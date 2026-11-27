# Migration from 0.1

Version 1.0 replaces the multi-robot workspace with one LW-v0 workstation. It is a
breaking API change. The previous workspace remains available at tag `v0.1.0`;
existing scripts cannot be used unchanged.

## Installation and assets

Use a fresh checkout and run `./install.sh`, or `uv sync --locked` if uv is already
installed. The old `packages/linker-sim` layout and asset submodule are no longer used.
LW-v0 is installed as `linker-lw-v0` from the Git release pinned in `pyproject.toml`
and `uv.lock`. Installation requires Git access to linker-sim-assets.

Use the source checkout for this release: gallery entries are read from its
`policy-gallery/` directory and are not included in the wheel.

## Application code

Replace backend, registry, environment and task setup with
`linker_sim.workstation.Station`. Configure its arm actions with `arm_action_cfg`
and its hand actions with `hand_action_cfg`. Pass commands to `Station.step` and
read joint state with `Station.observe`. See the [API reference](api.md) for
command shapes, units and frames, and [getting started](getting-started.md#5-your-first-script)
for a complete example.

The old Hydra robot/controller/task configurations and IK, PD and OSC controller
interfaces are removed. LW-v0 has two AR5 arms with O6 hands; the previous robot
combinations and asset composer are not exposed by this API.

The previous telemetry conversion, replay and recorder interfaces are removed.
There is no compatibility adapter or replacement telemetry replay command in 1.0.
Keep workloads that require those interfaces on `v0.1.0` until they are ported.

Use `linker-sim play` for manual control, `linker-sim gallery` for pretrained policies,
and `viewer.Panel` with `viewer.run` for an interactive application. The gallery's
weights have separate research-only terms; see its manifest and NOTICE.

## Current limits

The simulation runs one station on the CPU. Camera poses and intrinsics can be read,
but camera images are not simulated. Supported platforms and viewer requirements
are listed in [requirements](requirements.md).
