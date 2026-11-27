# Getting started

From nothing to the robot moving in your browser: install, check, play, the gallery,
your first script. It takes about ten minutes, most of it downloading.

You need a Mac with Apple Silicon or a Linux computer (x86_64 or ARM64), about 2 GB of
disk, `git`, and a browser. No GPU is needed for the simulation. See
[requirements](requirements.md) for details.

## 1. Install

Get the repository, then run its installer:

```bash
git clone https://github.com/linker-bot/linker-sim
cd linker-sim
./install.sh
```

`install.sh` installs uv (a Python package manager) into `~/.local/bin` if you don't have
it, then Python 3.12 and every package into `.venv` in this folder, exactly as pinned in
`uv.lock`. uv also stores managed Python installations and caches outside this folder,
and its installer may update shell startup files. At the end it runs
`linker-sim doctor` (next section).

If the installer installed uv, this terminal doesn't find it yet (`uv: command not found`):
open a new terminal, or run `source ~/.local/bin/env`.

LW-v0's robot files come with the installation: the package `linker-lw-v0` from
[linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) at its release tag
`v1.0.0`, which uv fetches with `git`. No `git submodule update` is needed.
Use this Git checkout for now: the wheel does not include the gallery entries.

## 2. Check

```bash
uv run linker-sim doctor
```

It checks the installed versions, runs Warp and torch on the CPU, and simulates LW-v0 for
one second. The last check reads something like:

```text
[ok] simulation: LW-v0 1.0.0, MuJoCo 4.0.1: home pose held to 9.7e-07 rad; 1 s simulated at 15x real time on one thread
All checks passed.
```

The real-time factor (here 15×) says how much faster than real time this computer runs
the station. Above about 4× everything should run at real time (an estimate until slower
computers are measured, see [requirements](requirements.md)).

Every command runs in the folder's environment through `uv run`. Or activate it once per
terminal with `source .venv/bin/activate`, then drop the `uv run`.

## 3. Play

```bash
uv run linker-sim play
```

It prints an address such as `http://127.0.0.1:8080/`. Open it in your browser: the
station appears after a few seconds, with a 5 cm box on the table under the right hand.
Studio's own messages in the terminal ("Model changed: the browser page reloads
automatically.", log lines about a browser connecting) are normal. **Click into the scene once**, so the page gets the
keys. Then:

| Key | What it does |
| --- | --- |
| I / K, J / L, U / O | Move the active arm's palm goal (the right arm at the start) forward / back, left / right, up / down, relative to your view |
| Shift + the same keys | Turn the goal instead |
| Tab | Switch arm |
| 1 / 2 / 3 / 4 | Hand: open / close / pinch / thumb oppose |
| R | Reset |
| Mouse | Turn and move the camera |

The arm follows its goal as the real arm does, within the real controller's limits. The
panel shows the arm's tool pose, how far the palm still is from the goal, how often a
limit intervened, the contacts and the speed. Ctrl+C in the terminal stops it.

The same playground in the other viewer:

```bash
uv run linker-sim play --viewer viser
```

In viser each key press moves the goal a little (holding a key repeats it), and you can
also drag the goal's handle with the mouse. The [viewers guide](viewers.md) explains the
differences.

**Working on a remote Linux computer?** The page is served to that computer only. Forward
the port: `ssh -L 8080:127.0.0.1:8080 <computer>`, then open
`http://127.0.0.1:8080/` on yours.

## 4. The gallery

Pretrained policies running on LW-v0:

```bash
uv run linker-sim gallery list
uv run linker-sim gallery info ycb-grasp
uv run linker-sim gallery run ycb-grasp            # or --viewer viser
```

The first run downloads `policy.npz` and `objects.npz` from the URLs in
`policy-gallery/ycb-grasp/manifest.yaml`. Files are checked against the manifest's
SHA-256 values and cached under `${XDG_CACHE_HOME:-~/.cache}/linker-lab/gallery/ycb-grasp/`.
The first lesson uses these files too. If a release requires authentication, use the
URL and destination in the error message to download the files with an authenticated
client; subsequent runs reuse files whose checksums match.

In `ycb-grasp`, choose one of 31 objects in the panel's list and press **Drop**: it lands at
a random spot on the left half of the table. Once it rests there, press **Grasp**. The left arm and hand pick it up and hold it 15 cm
up, and the panel says whether it was grasped. **Reset** clears the table.

**Its weights are for research use only (non-commercial).** `gallery info` shows the
licences. A retrained policy that may be shipped in products comes with a later version.

## 5. Your first script

linker-sim is a Python library first; the playground and the gallery are built on it.
Save this as `first.py` in the folder and run `uv run python first.py`:

```python
import numpy as np

from linker_sim import hand
from linker_sim.workstation import Station, arm_action_cfg

station = Station(arm_left=arm_action_cfg("left", command="pose", frame="palm"))
pos, quat = station.palm_pose("left")  # the palm point, station frame
goal = np.concatenate(station.to_base("left", pos + (0, 0, 0.05), quat))  # 5 cm up
for _ in range(60):  # one simulated second: 60 decisions
    station.step(arm_left=goal, hand_left=hand.PRESETS["close"])
print("palm moved up", station.palm_pose("left")[0][2] - pos[2], "m")
```

Then build a lesson with buttons in a viewer: [a first lesson](lesson.md). Everything you
can use is in the [API reference](api.md).

## When something goes wrong

| What you see | What to do |
| --- | --- |
| `install.sh: linker-sim needs a Mac with Apple Silicon` | Intel Macs aren't supported (MuJoCo 4 has no build for them) |
| `glibc ... is too old` | Use a newer Linux: Ubuntu 20.04+ on x86_64, 22.04+ on ARM64 |
| A gallery download fails | The message names the file's URL and where it goes: download it by hand to that path (its sha256 is checked) |
| `doctor` fails on the simulation, or reports below 1× | The computer is too slow or busy; close other programs and try again, then report it |
| The page stays blank, or Studio says "Connecting…" | Wait up to a minute (Studio downloads the whole model, about 290 MB); try `--viewer viser` |
| The page is very slow, Studio's keys lag by seconds | The browser draws without a GPU: check `chrome://gpu`, or use `--viewer viser` |
| Keys do nothing | Click into the scene first. In Studio another tab may hold the panel: close it, or press "Steal control" |
| `Port N is already in use` (Studio, after `--port N`) | Pick another port, or leave out `--port` to take the first free one from 8080. viser moves to the next free port by itself; read the address it prints |

To report a problem, include the output of `uv run linker-sim doctor`.
