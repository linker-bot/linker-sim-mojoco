"""Rigid objects for LW-v0: import from meshes, contact material, placement on the table.

An object is one rigid body made of one or more mesh *parts* (for example the pieces of a
convex decomposition, or a single mesh). Each part collides as its exact convex hull. The
object's mass, centre of mass and inertia are given to MuJoCo explicitly; if you give only
the mass, they are computed for a uniform density over the parts' hulls (what collides).
Every collider gets the object contact material below.

    from linker_sim import objects
    from linker_sim.workstation import Station
    parts = objects.load_mesh("004_sugar_box.stl")
    box = objects.object_cfg(parts, mass=0.5, pos=objects.on_table(parts, (0.35, 0.0)))
    station = Station(objects={"sugar_box": box})

``object_cfg`` describes a free object; ``fixed_cfg`` a fixed one (no joint, it does not
move: a shelf, a bin, the floor) with the same material. ``Station`` adds them to the scene
as MuJoCo bodies. Poses are in the station frame (the arm-base frame of LW-v0: x forward
over the table, z up); quaternions are x, y, z, w, as in linker-lab.

Copied from linker-lab v1.0.0:src/linker_lab/objects.py: ``MATERIAL``, ``load_mesh``,
``mass_properties``, the inertia check, ``table`` and ``on_table``; change them there
first. New here: the configs are plain descriptions, and ``_add_to_spec`` builds the
MuJoCo body instead of a USD file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from linker_sim import assets

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["object_cfg", "fixed_cfg", "load_mesh", "mass_properties", "on_table", "table", "ObjectCfg", "Part", "MATERIAL"]

Part = tuple[np.ndarray, np.ndarray]
"""One mesh part: vertices (N, 3) in metres in the object frame, triangle indices (M, 3)."""

MATERIAL = {
    "friction": 1.0,
    "torsional_friction": 0.005,
    "rolling_friction": 0.0,
    "condim": 4,
    "solref": (0.004, 1.0),
    "solimp": (0.9, 0.95, 0.001, 0.5, 2.0),
    "margin": 0.0,
    "gap": 0.0001,
}
"""Contact settings of every object surface (MuJoCo meanings; solref in seconds).

Where an object touches the robot, the engine uses the larger friction and condim of the
two surfaces: hand-object contacts run with friction 4 (the hand's), condim 4 and this
torsional friction.
"""


@dataclass(frozen=True, eq=False)
class ObjectCfg:
    """An object to add to a :class:`~linker_sim.workstation.Station`; make it with
    :func:`object_cfg` or :func:`fixed_cfg`."""

    parts: tuple[Part, ...]
    mass: float | None
    """[kg]; None for a fixed object."""
    com: np.ndarray | None
    """Centre of mass [m] in the object frame; None for a fixed object."""
    inertia: np.ndarray | None
    """3x3 inertia about the centre of mass [kg m^2]; None for a fixed object."""
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    """Initial pose in the station frame; quaternion x, y, z, w."""
    visual: Part | None
    """A mesh shown instead of the parts (it never collides); None shows the parts."""


def load_mesh(path: str | Path, scale: float = 1.0) -> list[Part]:
    """Read a mesh file (STL, OBJ, PLY, ...) as a single part. Units: metres after ``scale``."""
    import trimesh

    mesh = trimesh.load_mesh(Path(path), force="mesh")
    return [(np.asarray(mesh.vertices, dtype=np.float64) * scale, np.asarray(mesh.faces))]


def mass_properties(parts: Sequence[Part], mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and 3x3 inertia about it [kg m^2], uniform density over the parts' hulls."""
    import trimesh

    hulls = [trimesh.convex.convex_hull(np.asarray(v, dtype=np.float64)) for v, _ in parts]
    volumes = np.array([h.volume for h in hulls])
    if np.any(volumes <= 0):
        raise ValueError("every part needs a convex hull with positive volume")
    density = mass / volumes.sum()
    com = sum(h.volume * h.center_mass for h in hulls) / volumes.sum()
    inertia = np.zeros((3, 3))
    for hull in hulls:
        m, d = density * hull.volume, hull.center_mass - com
        # moment_inertia is about the hull's own centre of mass for unit density
        inertia += density * hull.moment_inertia + m * (d @ d * np.eye(3) - np.outer(d, d))
    return com, inertia


def object_cfg(
    parts: Sequence[Part],
    mass: float,
    *,
    com: Sequence[float] | None = None,
    inertia: np.ndarray | None = None,
    pos: Sequence[float] = (0.35, 0.0, 0.0),
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    visual: Part | None = None,
) -> ObjectCfg:
    """A free object at ``pos`` / ``rot`` (quaternion x, y, z, w) in the station frame.

    Without ``com`` and ``inertia`` they are computed from ``mass`` (:func:`mass_properties`).
    """
    if (com is None) != (inertia is None):
        raise ValueError("give both com and inertia, or neither")
    if com is None:
        com, inertia = mass_properties(parts, mass)
    com, inertia = np.asarray(com, dtype=np.float64), np.asarray(inertia, dtype=np.float64)
    principal = np.linalg.eigvalsh(inertia)
    a, b, c = principal
    if a <= 0 or a + b < c:  # eigvalsh sorts ascending, so this covers every triangle inequality
        raise ValueError(f"not a physical inertia (principal moments {principal})")
    return ObjectCfg(_parts(parts), float(mass), com, inertia, _vec(pos, 3), _vec(rot, 4), visual)


def fixed_cfg(
    parts: Sequence[Part],
    *,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> ObjectCfg:
    """A fixed object at ``pos`` / ``rot`` (x, y, z, w): its parts collide with the object
    material but never move."""
    return ObjectCfg(_parts(parts), None, None, None, _vec(pos, 3), _vec(rot, 4), None)


@cache
def table() -> dict:
    """The table's collision box in the environment frame: ``top`` height, ``center`` and
    ``half_size`` (x, y) [m], read from LW-v0's support model."""
    root = ET.parse(assets.part_file("support")).getroot()
    geom = root.find(".//geom[@name='table_body']")
    pos = [float(v) for v in geom.get("pos").split()]
    size = [float(v) for v in geom.get("size").split()]
    support = assets.part_pose("support")
    if any(support["pos"]) or support["rot"] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("the support is expected at the origin of the environment")
    return {"top": pos[2] + size[2], "center": tuple(pos[:2]), "half_size": tuple(size[:2])}


def on_table(
    parts: Sequence[Part],
    xy: Sequence[float],
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    clearance: float = 0.001,
) -> tuple[float, float, float]:
    """Position that puts the object, turned by ``rot`` (x, y, z, w), ``clearance`` above the
    table top at ``xy``. Released there it drops onto the table."""
    lowest = min(float((np.asarray(v) @ _rotation(rot).T)[:, 2].min()) for v, _ in parts)
    return (float(xy[0]), float(xy[1]), table()["top"] + clearance - lowest)


def _rotation(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64) / np.linalg.norm(quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _parts(parts: Sequence[Part]) -> tuple[Part, ...]:
    if not parts:
        raise ValueError("an object needs at least one part")
    return tuple(
        (np.asarray(v, dtype=np.float64).reshape(-1, 3), np.asarray(f, dtype=np.int64).reshape(-1, 3))
        for v, f in parts
    )


def _vec(values: Sequence[float], n: int) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if len(out) != n:
        raise ValueError(f"expected {n} values, got {len(out)}")
    return out


def _add_to_spec(spec, name: str, cfg: ObjectCfg) -> None:
    """Add the object to a MuJoCo ``MjSpec`` as body ``name``: a free joint (unless fixed),
    the explicit mass properties, one convex mesh geom per part with :data:`MATERIAL`, and
    the visual mesh.

    Geom groups follow LW-v0's files: shown geoms are in group 1, hidden colliders in group 3
    (viewers hide it). Without a visual mesh the parts are shown (group 1) and collide.
    """
    import mujoco

    x, y, z, w = cfg.rot
    body = spec.worldbody.add_body(name=name, pos=list(cfg.pos), quat=[w, x, y, z])
    if cfg.mass is not None:
        body.add_freejoint(name=name)
        body.explicitinertial = True
        body.mass = cfg.mass
        body.ipos = cfg.com.tolist()
        i = cfg.inertia
        body.fullinertia = [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]
    m = MATERIAL
    for k, (vertices, faces) in enumerate(cfg.parts):
        spec.add_mesh(
            name=f"{name}/part_{k}",
            uservert=vertices.ravel().tolist(),
            userface=faces.ravel().tolist(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
        )
        body.add_geom(
            name=f"{name}/part_{k}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{name}/part_{k}",
            friction=[m["friction"], m["torsional_friction"], m["rolling_friction"]],
            condim=m["condim"],
            solref=list(m["solref"]),
            solimp=list(m["solimp"]),
            margin=m["margin"],
            gap=m["gap"],
            group=1 if cfg.visual is None else 3,
        )
    if cfg.visual is not None:
        vertices, faces = cfg.visual
        spec.add_mesh(
            name=f"{name}/visual",
            uservert=np.asarray(vertices, dtype=np.float64).ravel().tolist(),
            userface=np.asarray(faces, dtype=np.int64).ravel().tolist(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,  # any surface; mass comes from above
        )
        body.add_geom(
            name=f"{name}/visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{name}/visual",
            contype=0,
            conaffinity=0,
            group=1,
        )
