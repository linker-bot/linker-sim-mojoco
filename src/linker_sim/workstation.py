"""LW-v0 on the CPU: the station, its actions, frames and observations.

A :class:`Station` is one LW-v0 workstation simulated by MuJoCo: LW-v0's scene
``lw_v0.xml`` (``linker_sim.assets``), loaded unchanged (the two arm units, the top camera
and the support at their poses, the solver options of ``linker_sim.physics``, the home
keyframe), plus a floor and any objects you add, and one action per arm and hand (``linker_sim.arm``,
``linker_sim.hand``), as linker-lab's ``LWv0EnvCfg``:

    from linker_sim import hand, objects
    from linker_sim.workstation import Station, arm_action_cfg
    station = Station(objects={"box": objects.object_cfg(parts, 0.1, pos=...)},
                      arm_left=arm_action_cfg("left", command="pose", frame="palm"))
    station.step(hand_left=hand.PRESETS["close"])   # one decision: 8 physics steps, 1/60 s
    joints = station.observe()          # 13 joint positions and velocities per unit
    pos, quat = station.palm_pose("left")

The station frame is the arm-base frame: x forward over the table, z up; the table top is
0.5 m below the origin. Quaternions are x, y, z, w, as in linker-lab. Pose commands are
given in each arm's own base frame instead (the frame the real controller uses; see
:meth:`Station.to_base`).

Reading ``station.model`` and ``station.data`` (MuJoCo's ``MjModel`` and ``MjData``) is
allowed for anything this module lacks. Writing to them is not supported: it bypasses the
drive models and may change in any release.

Joint, body and frame names, :func:`arm_action_cfg` and :func:`hand_action_cfg` are
linker-lab's (copied from linker-lab v1.0.0:src/linker_lab/workstation.py; change them
there first). In MuJoCo every name of an arm unit is prefixed with its side, e.g.
``left/hand_lh_hand_base_link``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mujoco
import numpy as np
import torch

from linker_sim import arm, assets, hand, objects, physics

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["Station", "RigidObject", "arm_action_cfg", "hand_action_cfg", "SIDES", "ARM_JOINTS", "HAND_JOINTS", "TCP_BODY", "PALM_BODY", "PALM_OFFSET", "HOME", "FLOOR_TOP"]

SIDES = ("left", "right")

ARM_JOINTS = {side: assets.controller(side)["joints"] for side in SIDES}
"""The seven arm joints J1..J7 of each arm unit (``arms/ar5_08/controller.yaml``)."""

HAND_JOINTS = {side: assets.motor(side)["shafts"] for side in SIDES}
"""The six driven hand joints (thumb pitch, thumb yaw, index, middle, ring, pinky;
``hands/o6/motor.yaml``).

The other five hand joints (thumb ip and the four dip joints) follow them through fixed
couplings in LW-v0's files."""

TCP_BODY = {side: f"arm_AR5_5_08{side[0].upper()}_W4C4A6_tcp" for side in SIDES}
"""The arm's tool point (flange face), the frame the real controller reports."""

PALM_BODY = {side: f"hand_{side[0]}h_hand_base_link" for side in SIDES}
PALM_OFFSET = (0.055, 0.0, 0.055)
"""The palm point in the hand's base-link frame [m], the same on both hands.

The base frame's x axis is the palm normal (fingers flex towards +x), z points from the
wrist to the knuckles. The palm face is at x = 0.02, so the point lies 3.5 cm in front of
the palm, halfway between wrist and knuckles."""

HOME = {side: assets.home(side) for side in SIDES}
"""The home pose [rad] per side and joint (all 18 joints of each unit), the keyframe ``home``
of ``lw_v0.xml``: palms down about 0.3 m above the table, hands open, thumbs rotated in."""

FLOOR_TOP = -1.267
"""Height of the floor [m] in the station frame: the foot of the support (the lowest point
of its visual mesh; the table's collision box ends 3 mm lower). Copied from linker-lab
v1.0.0:src/linker_lab/playground.py."""

FLOOR_SIZE = (6.0, 6.0, 0.05)
"""The floor is a fixed box [m] under the table's centre, with the object material."""


def arm_action_cfg(side: str, *, frame: str = "tcp", **options) -> arm.ArmActionCfg:
    """An :class:`~linker_sim.arm.ArmActionCfg` for one arm of LW-v0.

    ``options`` are the action's options (``command``, ``interpolation``, ``stream``,
    ``limits``, ``ik_damping``). Pose commands (``command="pose"``) move ``frame``: ``"tcp"``,
    the arm's tool point (the frame the real controller reports), or ``"palm"``, the palm
    point of the hand (:data:`PALM_OFFSET`). E.g.
    ``Station(arm_left=arm_action_cfg("left", command="pose", frame="palm"))``.
    """
    bodies = {"tcp": (TCP_BODY[side], (0.0, 0.0, 0.0)), "palm": (PALM_BODY[side], PALM_OFFSET)}
    if frame not in bodies:
        raise ValueError(f"frame must be 'tcp' or 'palm', not {frame!r}")
    body, offset = bodies[frame]
    return arm.ArmActionCfg(
        asset_name=side, joint_names=list(ARM_JOINTS[side]), body_name=body, body_offset=offset, **options
    )


def hand_action_cfg(side: str, **options) -> hand.HandActionCfg:
    """A :class:`~linker_sim.hand.HandActionCfg` for one hand of LW-v0 (``options``: e.g.
    ``normalized=True``)."""
    motor = assets.motor(side)
    return hand.HandActionCfg(asset_name=side, joint_names=list(motor["shafts"]), motor=motor, **options)


def floor_cfg() -> objects.ObjectCfg:
    """The floor as a fixed object: its top at :data:`FLOOR_TOP`."""
    hx, hy, hz = (s / 2 for s in FLOOR_SIZE)
    corners = np.array([(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
    faces = [
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
        (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ]  # fmt: skip
    center = objects.table()["center"]
    return objects.fixed_cfg([(corners, np.array(faces))], pos=(*center, FLOOR_TOP - hz))


class RigidObject:
    """A free object of a station, for reading its state (``station.objects[name]``)."""

    def __init__(self, station: Station, name: str):
        self.name = name
        self._station = station
        self._body = station.model.body(name).id
        joint = station.model.body_jntadr[self._body]
        self._qpos = station.model.jnt_qposadr[joint]
        self._dof = station.model.jnt_dofadr[joint]

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Position (3,) [m] and quaternion (4,) x, y, z, w of the object frame, station frame."""
        data = self._station.data
        w, x, y, z = data.xquat[self._body]
        return data.xpos[self._body].copy(), np.array([x, y, z, w])

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear velocity of the centre of mass [m/s] and angular velocity [rad/s] (3,),
        station frame (as linker-lab's ``root_com_vel_w``)."""
        out = np.zeros(6)
        mujoco.mj_objectVelocity(
            self._station.model, self._station.data, mujoco.mjtObj.mjOBJ_BODY, self._body, out, 0
        )
        return out[3:], out[:3]

    def set_pose(self, pos: Sequence[float], quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Put the object at a pose (station frame; quaternion x, y, z, w), at rest. :meth:`pose`
        and :meth:`velocity` read it at once; the next :meth:`Station.step` starts from it.

        The physics solver's starting guess for the object (MuJoCo's warm start) is cleared
        too, so what happens next depends only on where it was put, not where it came from.
        """
        model, data = self._station.model, self._station.data
        x, y, z, w = (float(v) for v in quat)
        norm = np.linalg.norm((w, x, y, z))
        data.qpos[self._qpos : self._qpos + 7] = (*(float(v) for v in pos), w / norm, x / norm, y / norm, z / norm)
        data.qvel[self._dof : self._dof + 6] = 0.0
        data.qacc_warmstart[self._dof : self._dof + 6] = 0.0
        # The poses and velocities that pose() and velocity() read (about 10 µs).
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)


class Station:
    """One LW-v0 workstation on the CPU.

    ``objects``: name -> :func:`linker_sim.objects.object_cfg` / ``fixed_cfg``, added to the
    scene as MuJoCo bodies of that name. ``floor``: a fixed floor at :data:`FLOOR_TOP`, so
    objects knocked off the table stay in view.

    ``arm_left``, ``arm_right``: :func:`arm_action_cfg` of each arm; by default joint goals
    [rad] through the trajectory generator, within the real limits. ``hand_left``,
    ``hand_right``: :func:`hand_action_cfg` of each hand; by default six shaft targets
    [rad]. The actions are in :attr:`actions` (e.g. ``station.actions["arm_left"].limit_hits``).

    The station starts at home (:meth:`reset`). Each :meth:`step` is one decision: the
    actions take their commands, then the physics advances 8 steps of 1/480 s.
    """

    def __init__(
        self,
        objects: Mapping[str, objects.ObjectCfg] | None = None,
        floor: bool = True,
        *,
        arm_left: arm.ArmActionCfg | None = None,
        arm_right: arm.ArmActionCfg | None = None,
        hand_left: hand.HandActionCfg | None = None,
        hand_right: hand.HandActionCfg | None = None,
    ):
        from linker_sim import objects as objects_module

        extras = dict(objects or {})
        if floor:
            if "floor" in extras:
                raise ValueError("'floor' is the floor's name; give the object another name")
            extras = {"floor": floor_cfg(), **extras}
        spec = mujoco.MjSpec.from_file(str(assets.path("lw_v0.xml")))
        for name, cfg in extras.items():
            objects_module._add_to_spec(spec, name, cfg)
        # The home keyframe holds the robot's joints; free objects start where they were put.
        key = spec.key("home")
        key.qpos = [
            *key.qpos,
            *(v for c in extras.values() if c.mass is not None for v in (*c.pos, c.rot[3], *c.rot[:3])),
        ]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.objects = {
            name: RigidObject(self, name) for name, cfg in extras.items() if cfg.mass is not None
        }
        self._home = self.model.key("home").id
        # Every drive is a position drive on one joint.
        self._drive_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        joints = [
            self.model.joint(f"{side}/{name}").id
            for side in SIDES
            for name in (*ARM_JOINTS[side], *HAND_JOINTS[side])
        ]
        self._observed_qpos = self.model.jnt_qposadr[joints]
        self._observed_qvel = self.model.jnt_dofadr[joints]
        self._palm = {side: self.model.body(f"{side}/{PALM_BODY[side]}").id for side in SIDES}
        self._base = {side: self.model.body_rootid[self._palm[side]] for side in SIDES}
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home)
        self.data.ctrl[:] = self.data.qpos[self._drive_qpos]
        mujoco.mj_forward(self.model, self.data)
        cfgs = {
            "arm_left": arm_left or arm_action_cfg("left"),
            "arm_right": arm_right or arm_action_cfg("right"),
            "hand_left": hand_left or hand_action_cfg("left"),
            "hand_right": hand_right or hand_action_cfg("right"),
        }
        for name, cfg in cfgs.items():
            kind, side = name.split("_")
            wanted = arm.ArmActionCfg if kind == "arm" else hand.HandActionCfg
            if not isinstance(cfg, wanted) or cfg.asset_name != side:
                raise ValueError(f"{name} needs {kind}_action_cfg({side!r}, ...), got {cfg!r}")
        self.actions: dict[str, arm.ArmAction | hand.HandAction] = {
            name: (arm.ArmAction if name.startswith("arm") else hand.HandAction)(cfg, self)
            for name, cfg in cfgs.items()
        }
        """The actions by name, in the order they are applied (linker-lab's): ``arm_left``,
        ``arm_right`` (:class:`~linker_sim.arm.ArmAction`), ``hand_left``, ``hand_right``
        (:class:`~linker_sim.hand.HandAction`), for reading their state."""
        self.reset()

    @property
    def time(self) -> float:
        """Simulated time since the last reset [s]."""
        return float(self.data.time)

    def reset(self) -> None:
        """Back to the home keyframe: robot at home, every drive targeting its joint's home
        angle, objects at their initial poses, at rest; every action holds home."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home)
        self.data.ctrl[:] = self.data.qpos[self._drive_qpos]
        mujoco.mj_forward(self.model, self.data)
        for action in self.actions.values():
            action.reset()

    def step(
        self,
        *,
        arm_left: Sequence[float] | None = None,
        arm_right: Sequence[float] | None = None,
        hand_left: Sequence[float] | None = None,
        hand_right: Sequence[float] | None = None,
    ) -> None:
        """One decision: each action takes its command, then the physics advances
        :data:`~linker_sim.physics.DECIMATION` steps (1/60 s), each action writing its drive
        targets before every step.

        A command is a sequence of floats as linker-lab's action takes it: for an arm with
        joint commands 7 positions [rad] (14 with ``stream="points_velocity"``, 112 with
        ``"samples"``), with pose commands a position [m] and a quaternion x, y, z, w in the
        arm's base frame (:meth:`to_base`); for a hand 6 shaft targets (:data:`HAND_JOINTS`
        order). A non-finite value keeps that value of the previous command.

        An omitted command repeats the previous one (after a reset: hold home). For a goal
        or a pose that holds it; a stream is the caller's own path, so send it every decision
        (a repeated point holds the arm there; repeated samples replay the same samples).
        """
        commands = {"arm_left": arm_left, "arm_right": arm_right, "hand_left": hand_left, "hand_right": hand_right}
        for name, action in self.actions.items():
            command = commands[name]
            if command is None:  # non-finite values keep the previous command (linker-lab's rule)
                command = torch.full((1, action.action_dim), torch.nan)
            else:
                command = torch.as_tensor(np.asarray(command, dtype=np.float32)).reshape(1, -1)
                if command.shape[1] != action.action_dim:
                    raise ValueError(f"{name} takes {action.action_dim} values, got {command.shape[1]}")
            action.process_actions(command)
        model, data = self.model, self.data
        for _ in range(physics.DECIMATION):
            for action in self.actions.values():
                action.apply_actions()
            mujoco.mj_step(model, data)
        # mj_step leaves body poses, velocities and Jacobians at the state before its last
        # integration. Bring them to the current joint positions and velocities, so what is read
        # (palm and object poses, the pose commands' Jacobian) is the state the step ended in.
        # The physics is unaffected: every step computes them anew.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

    def observe(self) -> np.ndarray:
        """Positions [rad] and velocities [rad/s] of the 13 driven joints of each unit (52,), as
        linker-lab's ``LWv0EnvCfg``: left positions, right positions, left velocities, right
        velocities; per unit :data:`ARM_JOINTS` then :data:`HAND_JOINTS`."""
        qpos, qvel = self.data.qpos[self._observed_qpos], self.data.qvel[self._observed_qvel]
        return np.concatenate((qpos, qvel))

    def palm_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Position (3,) [m] and quaternion (4,) x, y, z, w of the palm point of one arm unit,
        station frame. The orientation is the hand base frame's (see :data:`PALM_OFFSET`)."""
        body = self._palm[side]
        rotation = self.data.xmat[body].reshape(3, 3)
        w, x, y, z = self.data.xquat[body]
        return self.data.xpos[body] + rotation @ np.array(PALM_OFFSET), np.array([x, y, z, w])

    def to_base(self, side: str, pos, quat) -> tuple[np.ndarray, np.ndarray]:
        """A pose in the station frame (position (3,) [m], quaternion (4,) x, y, z, w) expressed
        in the base frame of one arm, the frame its pose commands are given in.

        Each arm's base frame is its mounting flange on the column (``<side>/arm_..._base``,
        the arm unit's pose in ``lw_v0.xml``): 9 cm to the side of the station origin and
        turned 90 degrees about x, as the real controller sees it. E.g. the palm point 5 cm
        higher: ``pos, quat = station.palm_pose("left"); station.to_base("left", pos + (0, 0,
        0.05), quat)``."""
        base = self._base[side]
        rotation = self.data.xmat[base].reshape(3, 3)
        x, y, z, w = quat
        inverse, local = np.empty(4), np.empty(4)
        mujoco.mju_negQuat(inverse, self.data.xquat[base])
        mujoco.mju_mulQuat(local, inverse, np.array([w, x, y, z], dtype=float))
        return rotation.T @ (np.asarray(pos, dtype=float) - self.data.xpos[base]), np.array([*local[1:], local[0]])
