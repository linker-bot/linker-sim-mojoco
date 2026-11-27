"""The arm and hand actions in a station: drive reference, pose commands, limits, the hand's
torque guard.

Acceptance checks: the drive reference replayed in
closed loop, a 5 cm palm-point move, and the hand pressing on stops within its torque guard
and opening again.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from linker_sim import arm, hand, physics
from linker_sim.workstation import (
    ARM_JOINTS,
    HAND_JOINTS,
    HOME,
    SIDES,
    Station,
    arm_action_cfg,
    hand_action_cfg,
)

REFERENCE = np.load(Path(__file__).with_name("data") / "drive_reference.npz")


def qpos(station, side, names):
    model = station.model
    return station.data.qpos[[model.jnt_qposadr[model.joint(f"{side}/{n}").id] for n in names]]


def home(side, names):
    return np.array([HOME[side][n] for n in names])


@pytest.mark.parametrize("env", [0, 1])
def test_drive_reference_replays_in_closed_loop(env):
    """linker-lab's recording, replayed through a station without limits: the left arm gets
    the recorded per-step references (``stream="samples"``), the right arm the recorded
    Hermite points with velocities, the hands the recorded shaft targets. The measured joints
    and the written targets stay within 5e-6 rad (arm) and 1e-5 rad (hand) of the recording
    until the hands start pressing, and the hand within 0.02 rad while they press.

    Measured (MuJoCo 4.0.1): arm 1.7e-6, hand 3.8e-6 rad. The recording was made by
    linker-lab's GPU engine in float32; MuJoCo computes in float64, so the trajectories part
    by about float32 precision. In the last phase (from 3 s) the thumb presses against the
    index until the thumb-yaw drive reaches its torque guard (4.28 s) and pushes with a
    constant 9 N·m: the thumb's position is then set by the contact, where linker-lab's engine
    (Newton's collision detection) and MuJoCo's differ: 0.014 rad by the end. (linker-lab on
    the GPU replays its own recording to 9.7e-4 rad there.)"""
    station = Station(
        floor=False,
        arm_left=arm_action_cfg("left", interpolation=False, stream="samples", limits=None),
        arm_right=arm_action_cfg("right", interpolation=False, stream="points_velocity", limits=None),
    )
    press = int(REFERENCE["press_from_step"])  # from here the hands press toward their torque guard
    assert all(REFERENCE[f"{s}/hand_guard"][env].any() for s in SIDES)  # and reach it
    deviation = {f"{part}_{x}{during}": 0.0 for part in ("arm", "hand") for x in ("q", "target") for during in ("", "_pressing")}
    for k in range(REFERENCE["left/hand_command"].shape[1]):
        steps = slice(8 * k, 8 * k + 8)
        station.step(
            arm_left=np.concatenate((REFERENCE["left/arm_q_ref"][env, steps].ravel(), REFERENCE["left/arm_v_ref"][env, steps].ravel())),
            arm_right=np.concatenate((REFERENCE["right/arm_endpoint_q"][env, k], REFERENCE["right/arm_endpoint_v"][env, k])),
            hand_left=REFERENCE["left/hand_command"][env, k],
            hand_right=REFERENCE["right/hand_command"][env, k],
        )
        t = 8 * (k + 1)
        for side in SIDES:
            for part, names in (("arm", ARM_JOINTS[side]), ("hand", HAND_JOINTS[side])):
                if t < REFERENCE[f"{side}/{part}_q"].shape[1]:
                    measured = qpos(station, side, names)
                    key = f"{part}_q" + ("_pressing" if t > press else "")
                    deviation[key] = max(deviation[key], np.abs(measured - REFERENCE[f"{side}/{part}_q"][env, t]).max())
                written = station.data.ctrl[station.actions[f"{part}_{side}"]._ctrl]
                key = f"{part}_target" + ("_pressing" if t - 1 >= press else "")
                deviation[key] = max(deviation[key], np.abs(written - REFERENCE[f"{side}/{part}_target"][env, t - 1]).max())
    print(deviation)
    assert REFERENCE["left/arm_guard"][env].any()  # the recording drives the arm into the torque guard
    assert deviation["arm_q"] < 5e-6 and deviation["arm_target"] < 5e-6, deviation
    assert deviation["hand_q"] < 1e-5 and deviation["hand_target"] < 1e-5, deviation
    assert deviation["arm_q_pressing"] < 5e-6 and deviation["arm_target_pressing"] < 5e-6, deviation
    assert deviation["hand_q_pressing"] < 0.02 and deviation["hand_target_pressing"] < 0.02, deviation

@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_palm_point_pose_command_settles(axis):
    """Each palm point commanded 5 cm along one axis of its arm's base frame comes within 2 mm
    in under 1 s (linker-lab on the GPU: 0.73 s; here 0.57-0.73 s) and ends within 0.1 mm."""
    station = Station(
        arm_left=arm_action_cfg("left", command="pose", frame="palm"),
        arm_right=arm_action_cfg("right", command="pose", frame="palm"),
    )
    goals = {}
    for side in SIDES:
        pos, quat = station.to_base(side, *station.palm_pose(side))
        pos[axis] += 0.05
        goals[side] = np.concatenate((pos, quat))
    settled = {}
    for k in range(180):
        station.step(arm_left=goals["left"], arm_right=goals["right"])
        for side in SIDES:
            error = np.linalg.norm(station.to_base(side, *station.palm_pose(side))[0] - goals[side][:3])
            if side not in settled and error < 0.002:
                settled[side] = (k + 1) / physics.DECISION_HZ
    assert all(settled.get(side, 9) < 1.0 for side in SIDES), settled
    assert error < 1e-4
    _, quat = station.to_base("left", *station.palm_pose("left"))
    assert abs(abs(np.dot(quat, goals["left"][3:])) - 1) < 1e-8  # orientation held
    assert all(station.actions[f"arm_{side}"].limit_hits.item() == 0 for side in SIDES)

def test_hand_reopens_after_pressing_on_stops():
    """From an open hand, the left hand's six driven joints stop at 40 % of their range (the
    motor keeps the full range); the hand closes onto the stops and presses, opens, turns the
    thumb yaw onto its stop and presses, and opens again. No drive is asked for more than its
    force limit less 1 N·m (the torque guard, motor model ``geared-shaft-v1``; the
    written targets are :func:`linker_sim.arm.torque_guard`'s), and every joint opens again.

    Without the guard, on MuJoCo 4 a drive held at its limit loses its implicit damping,
    oscillates at the limit, and the motor model, which reads that as a load, stops the
    shaft for good: the fingers stayed closed (linker-sim-assets issue #28)."""
    station = Station(floor=False)
    model, data = station.model, station.data
    action = station.actions["hand_left"]
    joints = [model.joint(f"left/{n}").id for n in HAND_JOINTS["left"]]
    dofs, caps = model.jnt_dofadr[joints], model.jnt_actfrcrange[joints, 1]
    for j in range(model.njnt):  # an open hand
        if model.joint(j).name.startswith("left/hand_"):
            data.qpos[model.jnt_qposadr[j]] = 0.0
    mujoco.mj_forward(model, data)
    action.reset()
    model.jnt_range[joints, 1] *= 0.4
    kp, kd = model.actuator_gainprm[action._ctrl, 0], -model.actuator_biasprm[action._ctrl, 2]
    gains = [torch.from_numpy(x[None].astype(np.float32)) for x in (kp, kd, caps)]
    columns = torch.arange(6, dtype=torch.int32)
    worst, guarded = -np.inf, 0
    for phase, seconds, shafts in (
        ("close", 8, hand.PRESETS["close"]),
        ("open", 6, hand.PRESETS["open"]),
        ("thumb yaw", 8, (0.0, 1.36, 0.0, 0.0, 0.0, 0.0)),
        ("open again", 6, hand.PRESETS["open"]),
    ):
        for _ in range(round(seconds * physics.DECISION_HZ)):
            station.step(hand_left=shafts)
            worst = max(worst, float((np.abs(data.qfrc_actuator[dofs]) - caps).max()))
            expected = arm.torque_guard(action._request, action._q, action._qd, *gains, columns)
            torch.testing.assert_close(action._guarded, expected, rtol=0, atol=0)
            guarded += int((action._guarded != action._request).any())
        if phase.startswith("open"):
            q = data.qpos[model.jnt_qposadr[joints]]
            assert np.abs(q).max() < 1e-3 and np.abs(data.qvel[dofs]).max() < 1e-3, (phase, q)
    print(f"largest drive torque minus force limit {worst:.4f} N·m; {guarded} decisions with a guarded step")
    assert guarded > 0  # the stops drive the hand into its guard
    assert worst <= -1 + 1e-3, worst

def test_joint_goals_move_within_the_real_limits():
    """A goal 0.3 rad away on every joint of the right arm (inside the ranges) and a wrist goal beyond the octagon
    on the left: the references keep the velocity limits, the joints follow them, the left
    wrist stays inside the octagon and its limit hits are counted."""
    station = Station()
    goal_right = home("right", ARM_JOINTS["right"]) + 0.3 * np.array([1, 1, 1, -1, 1, 1, 1])  # J4 near its limit
    goal_left = home("left", ARM_JOINTS["left"])
    goal_left[5:] = (1.2, 1.5)  # J6, J7 in the corner region outside the octagon
    vmax = np.array(arm.VELOCITY)
    WIDE = torch.full((7,), 10.0, dtype=torch.float64)  # joint ranges that do not matter
    peak_ref = peak_measured = outside = 0.0
    for _ in range(4 * physics.DECISION_HZ):
        station.step(arm_left=goal_left, arm_right=goal_right)
        model, data = station.model, station.data
        for side in SIDES:
            dofs = [model.jnt_dofadr[model.joint(f"{side}/{n}").id] for n in ARM_JOINTS[side]]
            peak_measured = max(peak_measured, (np.abs(data.qvel[dofs]) / vmax).max())
            peak_ref = max(peak_ref, (station.actions[f"arm_{side}"].reference[1].abs().numpy()[0] / vmax).max())
        q = torch.from_numpy(qpos(station, "left", ARM_JOINTS["left"]))[None]
        projected, _ = arm.project(q, -WIDE, WIDE)  # the nearest point in the octagon
        outside = max(outside, float((q - projected).norm()))
    assert peak_ref <= 1 + 1e-6 and peak_measured < 1.05, (peak_ref, peak_measured)
    np.testing.assert_allclose(qpos(station, "right", ARM_JOINTS["right"]), goal_right, atol=1e-4)
    assert outside < 1e-3
    assert station.actions["arm_left"].limit_hits.item() > 0 and station.actions["arm_right"].limit_hits.item() == 0


def test_omitted_and_non_finite_commands_keep_the_previous_one():
    station = Station()
    station.step(hand_left=hand.PRESETS["close"])
    for _ in range(60):
        station.step()  # the hand keeps closing toward the last target
    shaft = station.actions["hand_left"].shaft.numpy()[0]
    assert shaft[2] > 0.5
    np.testing.assert_allclose(station.actions["hand_right"].shaft.numpy()[0], home("right", HAND_JOINTS["right"]), atol=1e-6)
    goal = home("left", ARM_JOINTS["left"])
    goal[0] += 0.1
    station.step(arm_left=goal)
    station.step(arm_left=[np.nan] * 6 + [goal[6] + 0.1])  # J1..J6 keep their goals, J7 changes
    np.testing.assert_allclose(station.actions["arm_left"].raw_actions.numpy()[0], [*goal[:6], goal[6] + 0.1], rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="takes 7 values"):
        station.step(arm_right=[0.0] * 6)


def test_reset_brings_every_action_back_to_home():
    station = Station()
    goal = home("left", ARM_JOINTS["left"])
    goal[5:] = (1.2, 1.5)
    for _ in range(30):
        station.step(arm_left=goal, hand_right=hand.PRESETS["close"])
    station.reset()
    np.testing.assert_allclose(station.actions["arm_left"].raw_actions.numpy()[0], home("left", ARM_JOINTS["left"]), atol=1e-6)
    assert station.actions["arm_left"].limit_hits.item() == 0
    np.testing.assert_allclose(station.actions["hand_right"].shaft.numpy()[0], home("right", HAND_JOINTS["right"]))
    start = station.data.qpos.copy()
    for _ in range(30):
        station.step()
    assert np.abs(station.data.qpos - start).max() < 1e-4  # holds home again


def test_pose_frame_and_jacobian():
    """The commanded frame's pose is the palm point seen from the arm's base, and its Jacobian
    matches finite differences of that pose."""
    station = Station(arm_right=arm_action_cfg("right", command="pose", frame="palm"))
    action = station.actions["arm_right"]
    pos, quat = action._frame_pose()
    expected = station.to_base("right", *station.palm_pose("right"))
    np.testing.assert_allclose(pos, expected[0], atol=1e-12)
    assert abs(abs(np.dot(quat, expected[1])) - 1) < 1e-12
    assert np.allclose(action.raw_actions.numpy()[0], np.concatenate((pos, quat)), atol=1e-6)  # holds where it is
    jacobian = action._frame_jacobian()
    model, data = station.model, station.data
    eps = 1e-6
    for i, address in enumerate(action._qpos):
        data.qpos[address] += eps
        mujoco.mj_kinematics(model, data)
        moved, moved_quat = action._frame_pose()
        data.qpos[address] -= eps
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose((moved - pos) / eps, jacobian[:3, i], atol=1e-5)
        w = np.empty(3)
        q0, q1 = np.array([quat[3], *quat[:3]]), np.array([moved_quat[3], *moved_quat[:3]])
        inverse, difference = np.empty(4), np.empty(4)
        mujoco.mju_negQuat(inverse, q0)
        mujoco.mju_mulQuat(difference, q1, inverse)
        mujoco.mju_quat2Vel(w, difference, eps)
        np.testing.assert_allclose(w, jacobian[3:, i], atol=1e-5)


def test_to_base_is_the_arm_base_frame():
    """Each arm's base is its unit's pose in lw_v0.xml: 9 cm to the side, turned 90 degrees
    about x; a pose and its base-frame version describe the same point."""
    station = Station()
    for side, y in (("left", 0.09), ("right", -0.09)):
        pos, quat = station.to_base(side, np.array([0.0, y, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(pos, 0, atol=1e-12)
        angle = 2 * np.arccos(min(1.0, abs(quat[3])))
        assert angle == pytest.approx(np.pi / 2, abs=1e-3) and abs(quat[0]) == pytest.approx(np.sin(np.pi / 4), abs=1e-3)
        pos, _ = station.to_base(side, np.array([0.0, y, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        assert abs(abs(pos[1]) - 1) < 1e-3  # station z is the base's +-y


def test_action_configs_are_checked():
    with pytest.raises(ValueError, match="arm_left needs"):
        Station(arm_left=arm_action_cfg("right"))
    with pytest.raises(ValueError, match="hand_right needs"):
        Station(hand_right=hand_action_cfg("left"))
    with pytest.raises(ValueError, match="stream must be 'points'"):
        Station(arm_left=arm_action_cfg("left", stream="samples"))
    with pytest.raises(ValueError, match="frame must be"):
        arm_action_cfg("left", command="pose", frame="wrist")
    normalized = Station(hand_left=hand_action_cfg("left", normalized=True))
    normalized.step(hand_left=[1.0] * 6)
    upper = normalized.model.jnt_range[[normalized.model.joint(f"left/{n}").id for n in HAND_JOINTS["left"]], 1]
    np.testing.assert_allclose(normalized.actions["hand_left"].processed_actions.numpy()[0], upper, rtol=1e-6)
