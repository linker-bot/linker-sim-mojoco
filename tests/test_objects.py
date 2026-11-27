"""Objects: mass properties, placement, and what MuJoCo builds from a config.

The first five tests are copied from linker-lab v1.0.0:tests/test_objects.py.
"""

import mujoco
import numpy as np
import pytest
import trimesh

from linker_sim import objects
from linker_sim.workstation import Station


def box(size, offset=(0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(size)
    return mesh.vertices + np.asarray(offset), mesh.faces


def test_box_mass_properties():
    a, b, c, mass = 0.06, 0.04, 0.1, 0.3
    com, inertia = objects.mass_properties([box((a, b, c), (0.01, 0, 0))], mass)
    assert np.allclose(com, (0.01, 0, 0))
    expected = mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    assert np.allclose(inertia, expected)


def test_two_parts_add_up_like_one_body():
    halves = [box((0.05, 0.05, 0.05), (-0.025, 0, 0)), box((0.05, 0.05, 0.05), (0.025, 0, 0))]
    whole = [box((0.1, 0.05, 0.05))]
    for a, b in zip(objects.mass_properties(halves, 0.2), objects.mass_properties(whole, 0.2)):
        assert np.allclose(a, b)


def test_load_mesh(tmp_path):
    path = tmp_path / "box.stl"
    trimesh.creation.box((0.05, 0.05, 0.05)).export(path)
    (vertices, faces), = objects.load_mesh(path, scale=2.0)
    assert np.allclose(np.ptp(vertices, axis=0), 0.1) and faces.shape[1] == 3


def test_rejects_unphysical_inertia():
    with pytest.raises(ValueError, match="physical inertia"):
        objects.object_cfg([box((0.05, 0.05, 0.05))], 0.1, com=(0, 0, 0), inertia=np.diag([1e-4, 1e-4, 3e-4]))
    with pytest.raises(ValueError, match="both"):
        objects.object_cfg([box((0.05, 0.05, 0.05))], 0.1, com=(0, 0, 0))


def test_table_and_placement():
    table = objects.table()
    assert table["top"] == pytest.approx(-0.4998)  # 0.5 m below the arm-base origin
    parts = [box((0.04, 0.06, 0.1))]
    x, y, z = objects.on_table(parts, (0.3, 0.1), clearance=0.0)
    assert (x, y) == (0.3, 0.1) and z == pytest.approx(table["top"] + 0.05)
    # Lying on its side (90 degrees about x), the 6 cm edge is vertical.
    q = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))
    assert objects.on_table(parts, (0.3, 0.1), q, clearance=0.0)[2] == pytest.approx(
        table["top"] + 0.03
    )


def geoms_of(model, body):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]


def hull_vertices(model, geom):
    """Vertices of the convex hull MuJoCo collides the mesh geom with."""
    return int(model.mesh_graph[model.mesh_graphadr[model.geom_dataid[geom]]])


def test_mujoco_builds_what_the_config_says():
    """Mass, centre of mass, inertia, pose and contact settings reach MuJoCo unchanged; each
    part collides as its exact convex hull."""
    parts = [box((0.06, 0.04, 0.1), (0.0, 0.0, 0.01)), box((0.02, 0.02, 0.02), (0.0, 0.0, 0.07))]
    com = (0.001, -0.002, 0.02)
    inertia = np.array([[3e-4, 1e-5, 0], [1e-5, 3.5e-4, 0], [0, 0, 2e-4]])
    rot = (0.0, 0.0, np.sin(0.3), np.cos(0.3))
    cfg = objects.object_cfg(parts, 0.25, com=com, inertia=inertia, pos=(0.3, 0.1, -0.2), rot=rot)
    station = Station(objects={"thing": cfg}, floor=False)
    model, data = station.model, station.data
    body = model.body("thing").id
    assert model.body_mass[body] == pytest.approx(0.25)
    assert np.allclose(model.body_ipos[body], com)
    axes = np.zeros(9)
    mujoco.mju_quat2Mat(axes, model.body_iquat[body])
    axes = axes.reshape(3, 3)
    assert np.allclose(axes @ np.diag(model.body_inertia[body]) @ axes.T, inertia, atol=1e-12)
    assert model.body_gravcomp[body] == 0 and model.body_jntnum[body] == 1
    assert model.jnt_type[model.body_jntadr[body]] == mujoco.mjtJoint.mjJNT_FREE
    pos, quat = station.objects["thing"].pose()
    assert np.allclose(pos, (0.3, 0.1, -0.2)) and np.allclose(quat, rot)

    m = objects.MATERIAL
    geoms = geoms_of(model, body)
    assert len(geoms) == 2
    for g in geoms:
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_contype[g] == 1 and model.geom_conaffinity[g] == 1
        assert model.geom_group[g] == 1  # no visual mesh: the parts are shown
        assert np.allclose(model.geom_friction[g], (m["friction"], m["torsional_friction"], m["rolling_friction"]))
        assert model.geom_condim[g] == m["condim"]
        assert np.allclose(model.geom_solref[g], m["solref"])
        assert np.allclose(model.geom_solimp[g], m["solimp"])
        assert model.geom_margin[g] == m["margin"] and model.geom_gap[g] == pytest.approx(m["gap"])
        assert hull_vertices(model, g) == 8  # a box's hull: its corners
    # The parts are where the config put them: each mesh's vertices, placed in the world,
    # are the part's vertices moved by the object's pose.
    rotation = objects._rotation(rot)
    for g, (vertices, _) in zip(geoms, parts):
        mesh = model.geom_dataid[g]
        local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        expected = vertices @ rotation.T + (0.3, 0.1, -0.2)
        assert np.allclose(np.sort(world, axis=0), np.sort(expected, axis=0), atol=1e-6)


def test_default_mass_properties_are_uniform_density_over_the_hulls():
    parts = [box((0.06, 0.04, 0.1))]
    com, inertia = objects.mass_properties(parts, 0.3)
    station = Station(objects={"b": objects.object_cfg(parts, 0.3)}, floor=False)
    body = station.model.body("b").id
    assert np.allclose(station.model.body_ipos[body], com)
    assert np.allclose(np.sort(station.model.body_inertia[body]), np.sort(np.linalg.eigvalsh(inertia)))


def test_fixed_object_never_moves():
    """A fixed object has no joint: MuJoCo welds it to the world, with the object material."""
    shelf = objects.fixed_cfg([box((0.2, 0.2, 0.02))], pos=(0.35, 0.0, -0.3))
    station = Station(objects={"shelf": shelf}, floor=False)
    model = station.model
    body = model.body("shelf").id
    assert model.body_jntnum[body] == 0 and model.body_weldid[body] == 0
    assert model.nq == mujoco.MjModel.from_xml_path(str(objects.assets.path("lw_v0.xml"))).nq
    (g,) = geoms_of(model, body)
    assert model.geom_condim[g] == objects.MATERIAL["condim"]
    assert "shelf" not in station.objects
    for _ in range(10):
        station.step()
    assert np.allclose(station.data.xpos[body], (0.35, 0.0, -0.3))


def test_visual_mesh_is_shown_and_never_collides():
    """With ``visual``, viewers show that one mesh; the parts still collide, hidden."""
    parts = [box((0.02, 0.02, 0.02), (x, 0, 0)) for x in (-0.03, 0.0, 0.03)]
    visual = box((0.08, 0.02, 0.02))
    station = Station(objects={"v": objects.object_cfg(parts, 0.1, visual=visual)}, floor=False)
    model = station.model
    geoms = geoms_of(model, model.body("v").id)
    collides = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    shown = [g for g in geoms if model.geom_group[g] < 3]
    assert len(collides) == 3 and len(shown) == 1 and not set(collides) & set(shown)
    assert model.geom(shown[0]).name == "v/visual"
    assert model.body_mass[model.body("v").id] == pytest.approx(0.1)
