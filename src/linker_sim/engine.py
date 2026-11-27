"""What linker-sim runs on, and the checks ``linker-sim doctor`` runs.

linker-sim steps LW-v0 with MuJoCo's own C engine (the ``mujoco`` package), one world on
one CPU thread: no GPU, no patches. The versions that decide how LW-v0 behaves or is
shown are pinned exactly in ``pyproject.toml``:

- ``mujoco`` 4.0.1. 4.0 is new, and 4.0.0 had a solver bug that 4.0.1 fixed; every bump
  runs the acceptance checks first. linker-lab runs the same LW-v0 files on MuJoCo 3.17
  (its GPU engine requires it); the two agree to 2e-6 rad unless a drive is at its torque cap.
- ``warp-lang`` 1.17.0 on Warp's CPU device, for the arm's motion generator, torque guard
  and limits: the same kernels and Warp version as linker-lab.
- ``torch`` 2.11.0 (CPU build), as in linker-lab: the actions and policies are torch code.
- ``viser`` and ``mjviser``: the backup viewer, a young API.

Importing this module imports none of them.
"""

from __future__ import annotations

import importlib.metadata as metadata
import time
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_versions() -> list[Check]:
    """Every package linker-sim pins is installed at the pinned version."""
    from packaging.requirements import Requirement

    checks = []
    for line in metadata.requires("linker-sim") or []:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            checks.append(Check(req.name, False, f"not installed (need {req.specifier})"))
            continue
        ok = req.specifier.contains(installed, prereleases=True)
        checks.append(Check(req.name, ok, f"{installed}" + ("" if ok else f" (need {req.specifier})")))
    return checks


def check_warp() -> Check:
    """Warp compiles and runs a kernel on the CPU (the arm's kernels run there)."""
    import numpy as np
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING  # no banner

    @wp.kernel
    def _scale(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
        i = wp.tid()
        y[i] = wp.float64(2.0) * x[i]

    start = time.perf_counter()
    try:
        x = wp.array(np.arange(4.0), dtype=wp.float64, device="cpu")
        y = wp.zeros(4, dtype=wp.float64, device="cpu")
        wp.launch(_scale, dim=4, inputs=[x], outputs=[y], device="cpu")
        ok = bool(np.array_equal(y.numpy(), 2.0 * np.arange(4.0)))
    except Exception as error:  # a missing compiler or library shows up here
        return Check("Warp on the CPU", False, f"a CPU kernel failed: {error}")
    wall = time.perf_counter() - start
    return Check(
        "Warp on the CPU", ok, f"CPU kernel ran in {wall:.1f} s (the first run compiles, later runs reuse it)"
    )


def check_torch() -> Check:
    """torch loads and computes on the CPU (the actions and policies are torch code)."""
    try:
        import torch

        ok = float(torch.arange(4.0, dtype=torch.float64).sum()) == 6.0
    except Exception as error:  # a broken native library shows up here
        return Check("torch on the CPU", False, f"failed to load or compute: {error}")
    return Check("torch on the CPU", ok, f"{torch.get_num_threads()} threads")


def check_simulation(seconds: float = 1.0) -> Check:
    """Simulate LW-v0 for ``seconds`` with the arm and hand actions holding the home pose.

    Builds a :class:`~linker_sim.workstation.Station` (LW-v0's ``lw_v0.xml``, the floor, and
    the default actions) and steps it at 480 Hz on one thread. Passes when the home pose
    holds and the simulation runs at least at real time.
    """
    import mujoco
    import numpy as np

    from linker_sim import physics
    from linker_sim.workstation import Station

    station = Station()
    station.step()  # the arm's Warp kernels compile on first use (once per machine): not timed
    station.reset()
    home = station.data.qpos.copy()
    decisions = round(seconds * physics.DECISION_HZ)
    start = time.perf_counter()
    for _ in range(decisions):
        station.step()
    wall = time.perf_counter() - start
    drift = float(np.abs(station.data.qpos - home).max())
    speed = station.time / wall
    held = bool(np.isfinite(station.data.qpos).all()) and drift < 1e-4
    detail = (
        f"LW-v0 {metadata.version('linker-lw-v0')}, MuJoCo {mujoco.__version__}: home pose held to "
        f"{drift:.1e} rad; {station.time:g} s simulated at {speed:.0f}x real time on one thread"
    )
    if speed < 1:
        detail += "; below real time, so this computer cannot run LW-v0 as shipped"
    return Check("simulation", held and speed >= 1, detail)
