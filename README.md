# linker-sim

> **linker-sim has been rebuilt** (1.0): it now simulates one Linker workstation, LW-v0, on
> MuJoCo 4.0, the current major release of the MuJoCo physics engine. The previous
> multi-robot workspace (AR5, P7 and A7-lite arms; L6, O6, L25 and L30 hands; the asset
> composer and telemetry replay) is no longer maintained and remains available at tag
> [`v0.1.0`](https://github.com/linker-bot/linker-sim/tree/v0.1.0).

One Linker workstation, LW-v0 (two AR5 arms with O6 hands, a table and cameras), behaving
like the real robot, simulated with MuJoCo 4 on an ordinary CPU. No GPU needed.

It is the same LW-v0 as in linker-lab, the company's GPU simulator: the same robot files,
drive models, controller limits and physics settings. linker-sim runs one station at real time
on macOS (Apple Silicon) or Linux, and shows it in your browser. It is a Python library to
build lessons and products on, with two working examples built on it: a playground to
drive the robot by hand, and a gallery of pretrained policies.

The gallery's policy weights are for research use only (see [the gallery](docs/getting-started.md#4-the-gallery)).
[中文快速入门](docs/zh-CN/README.md)

```bash
git clone https://github.com/linker-bot/linker-sim && cd linker-sim
./install.sh                              # macOS (Apple Silicon) or Linux; installs everything into .venv
uv run linker-sim doctor                  # checks the installation and this computer's speed
uv run linker-sim play                    # drive LW-v0 in your browser (MuJoCo Studio)
uv run linker-sim gallery run ycb-grasp   # a pretrained grasp policy
```

```python
import numpy as np

from linker_sim import hand
from linker_sim.workstation import Station, arm_action_cfg

station = Station(arm_left=arm_action_cfg("left", command="pose", frame="palm"))
pos, quat = station.palm_pose("left")
goal = np.concatenate(station.to_base("left", pos + (0, 0, 0.05), quat))  # palm 5 cm up
for _ in range(60):  # one simulated second
    station.step(arm_left=goal, hand_left=hand.PRESETS["close"])
```

## Documentation

- [Getting started](docs/getting-started.md): install, check, play, the gallery, a first
  script, troubleshooting
- [A first lesson](docs/lesson.md): a small lesson built on the API, step by step
  ([`examples/pick_up_a_box.py`](examples/pick_up_a_box.py))
- [API reference](docs/api.md): everything you can build on, and the versioning rules
- [The two viewers](docs/viewers.md): MuJoCo Studio and viser, their differences, network
  access, embedding in your page
- [Requirements](docs/requirements.md): supported computers, measured speeds, how to test
  a computer
- [Migration from 0.1](docs/migration.md): removed features and the replacement API
- [Changelog](CHANGELOG.md)
- [中文快速入门](docs/zh-CN/README.md) (Chinese: install, play, the gallery, the first lesson)

## Scope and installation

Install from a Git checkout using `./install.sh` or `uv sync --locked`. The gallery
currently reads `policy-gallery/` from the checkout; the wheel does not include its
entries. Keep the checkout when using the playground, gallery and examples.

The robot assets are installed as the `linker-lw-v0` Python package from a Git URL.
No submodule initialization is required, but installation still requires `git` and
access to GitHub. The gallery and the first lesson download policy and object files
on first use, verify their SHA-256 checksums and cache them under
`${XDG_CACHE_HOME:-~/.cache}/linker-lab/gallery/`.

Camera poses and intrinsics are available through the API; simulated camera images
are not provided. Windows and Intel Macs are not supported. For changes from 0.1,
see [Migration from 0.1](docs/migration.md).

## Develop

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

- The tests are linker-sim's acceptance checks: LW-v0 loads unchanged, home holds, the
  drive reference replays, pose commands settle, the gallery grasps, both viewers work.
- They also check the docs: `docs/api.md` must list exactly each public module's
  `__all__`, and every Python example in the README and `docs/` must run.
- `tests/test_copies.py` is opt-in: set `LINKER_LAB_CHECKOUT` to a linker-lab checkout
  containing tag `v1.0.0` to verify the shared code.
- The viewer tests open both viewers in headless Chrome and skip without it; set `CHROME`
  to its path if it is not found.
- `LINKER_SIM_BROWSER_VIEWERS=viser` opens viser only. Use this on machines without a GPU, where Chrome
  draws LW-v0 too slowly for Studio's input. Run the Studio ones on a computer with a GPU
  before merging.
- The arm and hand models, the reading of LW-v0's files and the gallery's policy code are
  copied from linker-lab and must stay identical: `uv run python tools/check_copies.py <linker-lab
  checkout>` compares them.
