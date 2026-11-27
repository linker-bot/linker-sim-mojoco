"""LW-v0's files as linker-sim uses them: solver options, parts, cameras, the robot, and the
acceptance check "LW-v0 loads unchanged".

The tests down to ``test_closing_hand_stops_thumb_on_index_finger`` are copied from linker-lab
v1.0.0:tests/test_lw_v0.py; change them there first. The rest are linker-sim's: a station adds
a floor and objects to ``lw_v0.xml``; every body, joint, mass, drive, contact exclusion,
coupling and gravity compensation of LW-v0 must stay exactly as MuJoCo compiles ``lw_v0.xml``
alone, and the solver options must be those of ``linker_sim.physics``.
"""

import math

import mujoco
import numpy as np
import pytest
import trimesh

from linker_sim import assets, objects, physics
from linker_sim.workstation import ARM_JOINTS, HAND_JOINTS, SIDES, Station


def test_physics_values():
    """The values LW-v0 was trained and verified with, as read from lw_v0.xml."""
    assert (physics.PHYSICS_HZ, physics.DECISION_HZ, physics.DECIMATION) == (480, 60, 8)
    assert physics.SOLVER == {
        "integrator": "implicitfast", "solver": "Newton", "iterations": 30, "ls_iterations": 50,
        "cone": "elliptic", "impratio": 10.0, "multiccd": True,
    }
    assert physics.CAPACITY == {"nconmax": 256, "njmax": 512}


def test_parts():
    for part in assets.PARTS:
        assert assets.part_file(part).is_file()
    assert assets.part_pose("left")["pos"] == (0.0, 0.09, 0.0)
    assert assets.part_pose("right")["pos"] == (0.0, -0.09, 0.0)
    assert assets.part_pose("support") == {"pos": (0.0, 0.0, 0.0), "rot": (0.0, 0.0, 0.0, 1.0)}
    x, y, z, w = assets.part_pose("left")["rot"]  # turned -90 degrees about x
    assert (x, y, z) == pytest.approx((-math.sqrt(0.5), 0, 0), abs=1e-4) and w == pytest.approx(math.sqrt(0.5), abs=1e-4)


def test_home_pose():
    home = assets.home("left")
    assert len(home) == 18 and home["arm_AR5_5_08L_W4C4A6_joint_1"] == 1.472
    assert home["hand_lh_thumb_cmc_yaw"] == 1.3
    assert assets.home("right")["hand_rh_thumb_cmc_yaw"] == 1.3599


def test_cameras():
    assert assets.camera("left", "camera_rgb_optical") == {
        "resolution": (1280, 800), "fx": 596.8096551281035, "fy": 593.024387405096, "cx": 640.0, "cy": 400.0,
    }
    assert assets.camera("top_camera", "sensor_left_optical")["resolution"] == (1280, 720)


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(assets.path("lw_v0.xml")))
    return mujoco, model


def test_scene_options_match_physics_module(scene):
    mujoco, model = scene
    assert model.opt.timestep == pytest.approx(1 / physics.PHYSICS_HZ)
    assert model.opt.iterations == physics.SOLVER["iterations"]
    assert model.opt.ls_iterations == physics.SOLVER["ls_iterations"]
    assert model.opt.impratio == pytest.approx(physics.SOLVER["impratio"])
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_scene_robot(scene):
    mujoco, model = scene
    for side, letter in (("left", "L"), ("right", "R")):
        j7 = model.joint(f"{side}/arm_AR5_5_08{letter}_W4C4A6_joint_7")
        assert np.allclose(j7.range, (-math.pi / 2, math.pi / 2))
        caps = [
            model.jnt_actfrcrange[model.joint(f"{side}/arm_AR5_5_08{letter}_W4C4A6_joint_{i}").id][
                1
            ]
            for i in range(1, 8)
        ]
        assert caps == [120, 120, 120, 120, 100, 100, 100]
    assert model.nu == 2 * (7 + 6)
    for camera in ("left/camera_rgb_optical", "right/camera_depth_optical", "top_camera/sensor_left_optical"):
        model.camera(camera)  # exists, on its optical body


def test_home_pose_holds_still_and_touches_nothing(scene):
    mujoco, model = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for i in range(model.nu):  # each position drive targets its joint's home angle
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    start = data.qpos.copy()
    for _ in range(physics.PHYSICS_HZ):
        mujoco.mj_step(model, data)
    assert np.abs(data.qpos - start).max() < 1e-4
    assert data.ncon == 0
    for joint_id in range(model.njnt):
        low, high = model.jnt_range[joint_id]
        if model.jnt_limited[joint_id]:
            assert low <= start[model.jnt_qposadr[joint_id]] <= high


FINGERS = {
    "thumb": ("thumb_metacarpals_base2", "thumb_metacarpals", "thumb_distal"),
    "index": ("index_proximal", "index_distal"),
    "middle": ("middle_proximal", "middle_distal"),
    "ring": ("ring_proximal", "ring_distal"),
    "pinky": ("pinky_proximal", "pinky_distal"),
}


def finger_of(body):
    return next((f for f, links in FINGERS.items() if body.endswith(links)), None)


def test_within_a_unit_only_different_fingers_collide(scene):
    mujoco, model = scene
    excluded = {frozenset((int(s) >> 16, int(s) & 0xFFFF)) for s in model.exclude_signature}
    colliders = sorted(
        {int(model.geom_bodyid[g]) for g in range(model.ngeom) if model.geom_contype[g]}
    )
    for unit in ("left", "right"):
        bodies = [b for b in colliders if model.body(b).name.startswith(f"{unit}/")]
        colliding = set()
        for i, a in enumerate(bodies):
            for b in bodies[i + 1 :]:
                adjacent = model.body_parentid[b] == a or model.body_parentid[a] == b
                if frozenset((a, b)) not in excluded and not adjacent:
                    colliding.add((model.body(a).name, model.body(b).name))
        assert all(
            finger_of(a) and finger_of(b) and finger_of(a) != finger_of(b) for a, b in colliding
        ), sorted(colliding)[:3]
        assert len(colliding) == 3 * 8 + 6 * 2 * 2  # thumb x 4 fingers, finger x finger


def test_closing_hand_stops_thumb_on_index_finger(scene):
    """The thumb and index tip meet instead of passing through each other.

    The hand closes against the O6 motor's advancing torque (3 N m yaw, 5 N m flexion),
    standing in for the hand model of hand.py.
    """
    mujoco, model = scene
    data = mujoco.MjData(model)
    saved = model.jnt_actfrcrange.copy()
    close = {"thumb_cmc_pitch": 0.45, "thumb_cmc_yaw": 1.2, "index_mcp_pitch": 1.45}
    try:
        for j in range(model.njnt):
            name = model.joint(j).name
            if name.startswith("right/hand_rh_") and model.jnt_actfrclimited[j]:
                cap = 3.0 if name.endswith("thumb_cmc_yaw") else 5.0
                model.jnt_actfrcrange[j] = (-cap, cap)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for i in range(model.nu):
            joint = model.joint(model.actuator_trnid[i, 0]).name
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
            for suffix, value in close.items():
                if joint == f"right/hand_rh_{suffix}":
                    data.ctrl[i] = value
        geom = {
            b: next(
                g
                for g in range(model.ngeom)
                if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name == b
            )
            for b in ("right/hand_rh_thumb_distal", "right/hand_rh_index_distal")
        }
        thumb, index = geom.values()
        deepest = np.inf
        for _ in range(2 * physics.PHYSICS_HZ):
            mujoco.mj_step(model, data)
            deepest = min(deepest, mujoco.mj_geomDistance(model, data, thumb, index, 0.05, None))
        assert -0.003 < deepest < 0  # touching, at most 3 mm into each other (8 mm if passing)
    finally:
        model.jnt_actfrcrange[:] = saved


# ---------------------------------------------------------------- linker-sim: the station


@pytest.fixture(scope="module")
def station():
    box = trimesh.creation.box((0.06, 0.06, 0.06))
    parts = [(box.vertices, box.faces)]
    return Station(objects={"box": objects.object_cfg(parts, 0.1, pos=objects.on_table(parts, (0.35, 0.0)))})


def _names(model, kind, n):
    return [mujoco.mj_id2name(model, kind, i) for i in range(n)]


# LW-v0's elements keep their ids (the station's come after them), so arrays compare
# element by element.
COMPARED = {
    (mujoco.mjtObj.mjOBJ_BODY, "nbody"): [
        "body_parentid", "body_weldid", "body_jntnum", "body_pos", "body_quat", "body_ipos",
        "body_iquat", "body_mass", "body_inertia", "body_gravcomp", "body_contype",
        "body_conaffinity",
    ],
    (mujoco.mjtObj.mjOBJ_JOINT, "njnt"): [
        "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range",
        "jnt_actfrclimited", "jnt_actfrcrange", "jnt_stiffness", "jnt_margin",
    ],
    (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"): [
        "actuator_trntype", "actuator_trnid", "actuator_dyntype", "actuator_gaintype",
        "actuator_biastype", "actuator_gainprm", "actuator_biasprm", "actuator_ctrllimited",
        "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange",
    ],
    (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"): [
        "geom_type", "geom_bodyid", "geom_dataid", "geom_contype", "geom_conaffinity",
        "geom_condim", "geom_priority", "geom_solmix", "geom_solref", "geom_solimp",
        "geom_margin", "geom_gap", "geom_friction", "geom_size", "geom_pos", "geom_quat",
        "geom_group",
    ],
    (mujoco.mjtObj.mjOBJ_MESH, "nmesh"): ["mesh_vertnum", "mesh_facenum"],
    (mujoco.mjtObj.mjOBJ_EQUALITY, "neq"): ["eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp"],
}
DOF = ["dof_armature", "dof_damping", "dof_frictionloss", "dof_bodyid", "dof_jntid"]


def test_station_keeps_lw_v0_unchanged(scene, station):
    b, s = scene[1], station.model
    for (kind, count), fields in COMPARED.items():
        n = getattr(b, count)
        assert _names(s, kind, n) == _names(b, kind, n), kind
        for field in fields:
            assert np.array_equal(getattr(s, field)[:n], getattr(b, field)[:n]), field
    for field in DOF:
        assert np.array_equal(getattr(s, field)[: b.nv], getattr(b, field)), field
    assert np.array_equal(s.qpos0[: b.nq], b.qpos0)
    assert np.array_equal(s.key_qpos[0, : b.nq], b.key_qpos[0])
    pairs = lambda m: set(m.exclude_signature.tolist())  # noqa: E731
    assert pairs(s) == pairs(b) and b.nexclude == 410
    # What the station adds: the floor and the box, each one body after LW-v0's.
    assert _names(s, mujoco.mjtObj.mjOBJ_BODY, s.nbody)[b.nbody :] == ["floor", "box"]
    assert s.nu == b.nu and s.neq == b.neq


def test_solver_options_are_the_physics_module(station):
    opt = station.model.opt
    assert opt.timestep == pytest.approx(1 / physics.PHYSICS_HZ)
    assert opt.integrator == getattr(mujoco.mjtIntegrator, f"mjINT_{physics.SOLVER['integrator'].upper()}")
    assert opt.solver == getattr(mujoco.mjtSolver, f"mjSOL_{physics.SOLVER['solver'].upper()}")
    assert opt.iterations == physics.SOLVER["iterations"]
    assert opt.ls_iterations == physics.SOLVER["ls_iterations"]
    assert opt.cone == getattr(mujoco.mjtCone, f"mjCONE_{physics.SOLVER['cone'].upper()}")
    assert opt.impratio == physics.SOLVER["impratio"]
    multiccd = not opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
    assert multiccd == physics.SOLVER["multiccd"]
    assert (opt.disableflags, opt.enableflags) == (0, 0)  # nothing else switched
    assert (station.model.nconmax, station.model.njmax) == tuple(physics.CAPACITY.values())


def test_robot_facts(station):
    """Drives, caps, the wrist range and gravity compensation as LW-v0's files set them, in the
    station."""
    model = station.model
    assert model.nu == 2 * (7 + 6)
    for side in SIDES:
        arm = [model.joint(f"{side}/{j}").id for j in ARM_JOINTS[side]]
        assert model.jnt_actfrcrange[arm, 1].tolist() == [120, 120, 120, 120, 100, 100, 100]
        assert np.allclose(model.jnt_range[arm[6]], (-math.pi / 2, math.pi / 2))  # J7 +-90 deg
        hand = [model.joint(f"{side}/{j}").id for j in HAND_JOINTS[side]]
        assert model.jnt_actfrcrange[hand, 1].tolist() == [60, 10, 60, 60, 60, 60]
        driven = {model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)}
        assert {f"{side}/{j}" for j in (*ARM_JOINTS[side], *HAND_JOINTS[side])} <= driven
        moving = [b for b in range(model.nbody) if model.body(b).name.startswith(f"{side}/") and model.body_weldid[b]]
        assert len(moving) > 18 and all(model.body_gravcomp[moving] == 1)  # weightless robot
    assert model.body_gravcomp[model.body("box").id] == 0  # objects fall
