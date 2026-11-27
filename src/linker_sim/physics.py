"""The physics settings LW-v0 is simulated with.

Copied from linker-lab v1.0.0:src/linker_lab/physics.py (the values, read from
``lw_v0.xml``); change them there first. linker-lab turns them into Isaac Lab's solver
configuration. linker-sim needs no configuration: MuJoCo reads the same options from
``lw_v0.xml`` when the station loads it.

The robot's own properties (masses, joint limits, drives, collision shapes and the
contact settings of its surfaces) are in LW-v0's files (``linker_sim.assets``); the contact
settings of objects are in ``linker_sim.objects``. The time step and the solver settings are
read from the ``<option>`` and ``<size>`` of ``lw_v0.xml``: 480 Hz, implicit-fast, Newton
solver with 30 iterations and 50 line-search iterations, elliptic cone, impratio 10,
multi-CCD. Only the decision rate is linker-sim's (and linker-lab's) own.
"""

from __future__ import annotations

from linker_sim import assets

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["PHYSICS_HZ", "DECISION_HZ", "DECIMATION", "SOLVER"]

_option = assets.scene().find("option")
_size = assets.scene().find("size")
_flag = _option.find("flag")

PHYSICS_HZ = round(1 / float(_option.get("timestep")))
"""Physics steps per second (480). Collision detection runs every step."""
if 1 / PHYSICS_HZ != float(_option.get("timestep")):
    raise ValueError("lw_v0.xml's timestep is not 1/N s")

DECISION_HZ = 60
"""Actions per second; each action is held for PHYSICS_HZ // DECISION_HZ physics steps."""

DECIMATION = PHYSICS_HZ // DECISION_HZ
"""Physics steps per action (8)."""

SOLVER = {
    "integrator": _option.get("integrator"),
    "solver": _option.get("solver"),
    "iterations": int(_option.get("iterations")),
    "ls_iterations": int(_option.get("ls_iterations")),
    "cone": _option.get("cone"),
    "impratio": float(_option.get("impratio")),
    # MuJoCo runs multi-CCD unless <flag multiccd="disable"/>.
    "multiccd": _flag is None or _flag.get("multiccd", "enable") == "enable",
}
"""Solver settings (names as in MJCF ``<option>``)."""

CAPACITY = {"nconmax": int(_size.get("nconmax")), "njmax": int(_size.get("njmax"))}
"""Contacts and constraint rows reserved per world. They size MuJoCo-Warp's fixed buffers
in linker-lab; native MuJoCo grows its buffers as needed, so here they limit nothing."""
