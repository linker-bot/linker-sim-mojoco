# Requirements

## Supported computers

| | Supported |
| --- | --- |
| Operating system | macOS on Apple Silicon (M1 or newer); Linux on x86_64 (glibc 2.28 or newer, e.g. Ubuntu 20.04+) or ARM64 (glibc 2.34 or newer, e.g. Ubuntu 22.04+) |
| Not supported | Intel Macs: MuJoCo 4 and Warp publish no builds for them. Windows: not in this version; it will be added if users need it |
| GPU | Not needed for the simulation. The browser that shows a viewer should draw with a GPU (see below) |
| Disk | About 2 GB: the environment (1.4 GB on Linux), LW-v0's files (56 MB) and the gallery's files |
| Network | For the installation only: up to 250 MB of packages and about 30 MB of LW-v0's files (from linker-sim-assets on GitHub) |
| Software | None beforehand: `install.sh` installs uv, Python 3.12 and every package. It needs `git` (uv fetches LW-v0's files with it) and `curl` or `wget` |
| Browser | A current Chrome, Edge, Firefox or Safari with hardware-accelerated WebGL 2. Tested: Chrome |

## Speed

linker-sim simulates one LW-v0 station on **one CPU core**, so the speed of a single core
matters, not the number of cores. Everything runs at real time when the computer can run
it faster than real time. `linker-sim doctor` measures yours: it simulates one second with
the arm and hand actions in the loop and reports the real-time factor.

What has been measured so far:

| Computer | `doctor` | Playground and gallery in a viewer |
| --- | --- | --- |
| Our development computer (x86_64 Linux desktop) | 15× | Real time in both viewers; spare capacity 4–9× |
| GitHub's shared runner, Linux x86_64 | 13× | not measured |
| GitHub's shared runner, Linux ARM64 | 6× | not measured |
| GitHub's shared runner, macOS (Apple Silicon) | 12× | not measured |

More measurements from the development computer, without a viewer:
- a station holding home runs at 15× real time
- the arm following pose commands, 12×
- a scripted grasp, 10–11×
- the gallery's scene (ycb-grasp 0.2.0, 38 objects), 7.8–10.6×

The browser needs more than the simulation. LW-v0 has 1.5 million triangles. With a desktop GPU,
both viewers draw at 60 frames per second; integrated GPUs (Intel, Apple) have not been
measured yet. In software, both are slow, and Studio's
panel and keys with them (see the [viewers guide](viewers.md)).

## Minimum requirements

**Not set yet.** They will come from measurements on more computers, slower ones and ones
with Intel or Apple GPUs.

Until then, an estimate. On the development computer, the heaviest case (the gallery
grasping in viser) kept 4.2× spare capacity against `doctor`'s 15×: it needs about a
quarter of `doctor`'s speed. So a computer whose `doctor` reports about 4× or more should
run the playground and the gallery at real time in either viewer, given a browser that
draws with a GPU. This has not been checked on a slow computer.

## Testing a computer

To test a computer for the minimum requirements, follow the
[getting-started guide](getting-started.md), then record:

1. **The computer:** operating system and version, processor, memory, GPU, and browser
   with its version.
2. **The installation:** whether `./install.sh` finished, and how long it took.
3. **`uv run linker-sim doctor`:** its whole output; the last line gives the real-time
   factor.
4. **The playground in each viewer** (`uv run linker-sim play`, then
   `uv run linker-sim play --viewer viser`):
   - how many seconds until the robot appears in the browser
   - whether moving the goal with I/K/J/L/U/O and the hand with 1–4 works
   - the panel's last line after a minute of moving: "Speed: …x real time (spare capacity
     …x)"
5. **The gallery in each viewer** (`uv run linker-sim gallery run ycb-grasp`, then with
   `--viewer viser`): choose an object, press Drop, then Grasp. Record whether it is
   grasped and the panel's last line while grasping.
6. **The lesson** (`uv run python examples/pick_up_a_box.py --check`): whether it passed (it
   downloads the gallery's policy on first use).
7. Anything that failed or looked wrong, with the messages in the terminal and in the
   browser's console (F12 in Chrome).
