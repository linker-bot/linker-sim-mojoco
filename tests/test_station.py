"""The station: home pose, a dropped box, the palm point, observations, reset and the floor.

The first three are acceptance checks: home holds, a box dropped
on the table comes to rest within 1 mm, and the palm point equals linker-lab's to 1e-4 m.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from linker_sim import assets, objects, physics
from linker_sim.workstation import (
    ARM_JOINTS,
    FLOOR_TOP,
    HAND_JOINTS,
    HOME,
    PALM_BODY,
    PALM_OFFSET,
    SIDES,
    Station,
)

BOX = 0.06  # edge [m], as linker-lab's GPU check
MASS = 0.1  # [kg]
XY = (0.35, 0.0)


def box_parts(edge=BOX):
    box = trimesh.creation.box((edge, edge, edge))
    return [(box.vertices, box.faces)]


def run(station, seconds):
    for _ in range(round(seconds * physics.DECISION_HZ)):
        station.step()


def hold_home(station, seconds=2.0) -> dict:
    """Step the station from home for ``seconds``; the largest deviation from home [rad] of
    its arm joints and of its hand joints (driven and coupled) during that time."""
    model = station.model
    arm = model.jnt_qposadr[[j for j in range(model.njnt) if "/arm_" in model.joint(j).name]]
    hand = model.jnt_qposadr[[j for j in range(model.njnt) if "/hand_" in model.joint(j).name]]
    assert len(arm) == 14 and len(hand) == 22
    home, peak = station.data.qpos.copy(), {"arm": 0.0, "hand": 0.0}
    for _ in range(round(seconds * physics.DECISION_HZ)):
        station.step()
        deviation = np.abs(station.data.qpos - home)
        peak = {"arm": max(peak["arm"], deviation[arm].max()), "hand": max(peak["hand"], deviation[hand].max())}
    return peak


def test_home_pose_holds():
    """For 2 s every arm joint stays within 1e-6 rad of home and every hand joint within
    1e-4 rad, and the robot touches nothing.

    The arm holds to about 3e-10 rad. The hand's coupled finger joints jitter by up to 2e-5
    rad (about 1 um at a fingertip): MuJoCo's default solver tolerance (1e-8; lw_v0.xml sets
    none) ends the solve after one or two iterations, and what remains of the couplings'
    constraint error moves the light finger links. It never settles, and its phase differs
    between processors. A tolerance of 1e-12 shrinks it to 2e-7 rad, so it is solver
    accuracy, not LW-v0's physics."""
    station = Station()
    peak = hold_home(station)
    assert station.time == pytest.approx(2.0)
    assert peak["arm"] < 1e-6 and peak["hand"] < 1e-4, peak
    assert station.data.ncon == 0


def test_box_dropped_on_the_table_comes_to_rest():
    """A 6 cm, 100 g box released 2 cm above the table: after 2 s it lies on the table top
    within 1 mm, within 1 mm of where it was released, at rest, and home still holds."""
    parts = box_parts()
    release = objects.on_table(parts, XY, clearance=0.02)
    station = Station(objects={"box": objects.object_cfg(parts, MASS, pos=release)})
    peak = hold_home(station)
    pos, quat = station.objects["box"].pose()
    linear, angular = station.objects["box"].velocity()
    assert abs(pos[2] - BOX / 2 - objects.table()["top"]) < 1e-3
    assert np.linalg.norm(pos[:2] - XY) < 1e-3
    assert np.linalg.norm(linear) < 1e-3 and np.linalg.norm(angular) < 1e-2
    assert abs(abs(quat[3]) - 1) < 1e-6  # did not tip
    assert peak["arm"] < 1e-6 and peak["hand"] < 1e-4, peak


def test_set_pose_puts_an_object_at_rest_where_given():
    """set_pose: pose() and velocity() read the new pose at once, at rest; the next steps
    start from it (the box, put 2 cm above the table, lands there), and what follows does not
    depend on where it came from."""
    parts = box_parts()
    station = Station(objects={"box": objects.object_cfg(parts, MASS, pos=objects.on_table(parts, XY))})
    box = station.objects["box"]
    run(station, 0.3)
    target = np.array(objects.on_table(parts, (0.45, 0.1), clearance=0.02))
    turn = np.array([0.0, 0.0, np.sin(0.25), np.cos(0.25)])  # 0.5 rad about z
    box.set_pose(target, turn)
    pos, quat = box.pose()
    assert np.allclose(pos, target) and np.allclose(quat, turn)
    assert not np.concatenate(box.velocity()).any()
    run(station, 1.0)
    landed = box.pose()
    station.reset()
    run(station, 0.7)  # elsewhere, moving differently, then the same placement
    box.set_pose(target, turn)
    run(station, 1.0)
    assert abs(landed[0][2] - BOX / 2 - objects.table()["top"]) < 1e-3
    assert np.linalg.norm(landed[0][:2] - target[:2]) < 1e-3
    assert np.allclose(box.pose()[0], landed[0], atol=1e-9)


def test_velocity_is_the_centre_of_mass_velocity():
    """velocity(): the centre of mass's linear velocity (linker-lab's root_com_vel_w), not
    the object frame's: an off-centre object spinning about its frame's origin moves its
    centre of mass at omega x r."""
    parts = [(v + (0.02, 0.0, 0.0), f) for v, f in box_parts()]  # centre of mass 2 cm along x
    station = Station(objects={"box": objects.object_cfg(parts, MASS, pos=(0.4, 0.0, 0.0))})
    dof = station.model.jnt_dofadr[station.model.body_jntadr[station.model.body("box").id]]
    station.data.qvel[dof : dof + 6] = (0.0, 0.0, 0.0, 0.0, 0.0, 3.0)  # 3 rad/s about z, origin at rest
    mujoco.mj_kinematics(station.model, station.data)
    mujoco.mj_comPos(station.model, station.data)
    mujoco.mj_comVel(station.model, station.data)
    linear, angular = station.objects["box"].velocity()
    assert np.allclose(angular, (0.0, 0.0, 3.0)) and np.allclose(linear, (0.0, 0.06, 0.0))


def test_palm_point_equals_linker_labs():
    """Position and orientation of the palm point at home, against linker-lab's recording.
    Orientations are compared as rotations: q and -q are the same one (LW-v0 0.6.0 writes the
    hand mount as exactly pi, which flips the quaternion's sign against the recording's)."""
    reference = json.loads((Path(__file__).parent / "data" / "palm_reference.json").read_text())
    station = Station()
    for side in SIDES:
        pos, quat = station.palm_pose(side)
        assert np.abs(pos - reference[side]["pos"]).max() < 1e-4
        assert 1 - abs(np.dot(quat, reference[side]["quat_xyzw"])) < 1e-8


def test_palm_point_lies_in_front_of_the_palm():
    """The palm face is the hand base hull's +x face; flexing a finger moves its tip to +x.

    Copied from linker-lab v1.0.0:tests/test_workstation.py."""
    station = Station(floor=False)
    model, data = station.model, station.data
    for side in SIDES:
        palm = model.body(f"{side}/{PALM_BODY[side]}").id
        hull = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == palm and model.geom_contype[g])
        mesh = model.geom_dataid[hull]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        # mesh vertices are in the geom frame; express them in the body frame
        rotation = data.xmat[palm].reshape(3, 3).T @ data.geom_xmat[hull].reshape(3, 3)
        offset = data.xmat[palm].reshape(3, 3).T @ (data.geom_xpos[hull] - data.xpos[palm])
        local = vertices @ rotation.T + offset
        assert local[:, 0].max() == pytest.approx(PALM_OFFSET[0] - 0.035, abs=1e-3)
        assert local[:, 2].min() < PALM_OFFSET[2] < local[:, 2].max()

        tip = model.body(f"{side}/hand_{side[0]}h_index_distal").id
        before = data.xmat[palm].reshape(3, 3).T @ (data.xpos[tip] - data.xpos[palm])
        flexed = mujoco.MjData(model)
        flexed.qpos[:] = data.qpos
        joint = model.joint(f"{side}/hand_{side[0]}h_index_mcp_pitch")
        flexed.qpos[model.jnt_qposadr[joint.id]] += 1.0
        mujoco.mj_kinematics(model, flexed)
        after = flexed.xmat[palm].reshape(3, 3).T @ (flexed.xpos[tip] - flexed.xpos[palm])
        assert after[0] - before[0] > 0.02


def test_observation_is_linker_labs_joint_state():
    """52 values: left positions, right positions, left velocities, right velocities; per unit
    the 7 arm joints, then the 6 driven hand joints."""
    station = Station()
    obs = station.observe()
    names = [f"{side}/{j}" for side in SIDES for j in (*ARM_JOINTS[side], *HAND_JOINTS[side])]
    assert obs.shape == (52,)
    assert obs[:26].tolist() == [HOME[n.split("/")[0]][n.split("/")[1]] for n in names]
    assert not obs[26:].any()
    station.data.qvel[:] = np.arange(station.model.nv)  # mark every value to check the order
    station.data.qpos[:] = np.arange(station.model.nq)
    obs = station.observe()
    for i, name in enumerate(names):
        joint = station.model.joint(name).id
        assert obs[i] == station.model.jnt_qposadr[joint]
        assert obs[26 + i] == station.model.jnt_dofadr[joint]
    obs[:] = 0
    assert station.data.qpos.any()  # a copy, not a view


def test_reset_returns_to_home_and_the_initial_object_poses():
    parts = box_parts()
    release = objects.on_table(parts, XY, clearance=0.05)
    station = Station(objects={"box": objects.object_cfg(parts, MASS, pos=release)})
    start = station.data.qpos.copy()
    assert np.allclose(station.objects["box"].pose()[0], release)
    station.data.ctrl[0] += 0.1  # a drive moved away from home
    run(station, 0.5)
    assert not np.allclose(station.data.qpos, start)
    station.reset()
    assert station.time == 0.0
    assert np.array_equal(station.data.qpos, start) and not station.data.qvel.any()
    assert np.array_equal(station.data.ctrl, station.data.qpos[station.model.jnt_qposadr[station.model.actuator_trnid[:, 0]]])


def test_floor_is_at_the_foot_of_the_support():
    """The floor top is the lowest point of the support's visual mesh; a box dropped next to
    the table lands on it."""
    station = Station(objects={"box": objects.object_cfg(box_parts(), MASS, pos=(0.35, 0.9, -1.0))})
    model, data = station.model, station.data
    geom = model.geom("support/support_visual").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    lowest = (vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])[:, 2].min()
    assert FLOOR_TOP == pytest.approx(lowest, abs=5e-4)
    run(station, 1.0)
    assert station.objects["box"].pose()[0][2] - BOX / 2 == pytest.approx(FLOOR_TOP, abs=1e-3)
    assert "floor" not in station.objects  # fixed: nothing to read


def test_without_floor_and_name_clash():
    scene = mujoco.MjModel.from_xml_path(str(assets.path("lw_v0.xml")))
    assert Station(floor=False).model.nbody == scene.nbody
    with pytest.raises(ValueError, match="floor"):
        Station(objects={"floor": objects.fixed_cfg(box_parts())})
