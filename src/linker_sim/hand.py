"""The O6 hand: a self-locking motor model and the ``HandAction``.

The real O6 drives each of its six actuated joints (thumb pitch, thumb yaw, index,
middle, ring, pinky) with a motor behind a self-locking gear: the motor shaft moves
toward its target, slowly and only as far as its torque allows, and an external load
can stop it but never push it back. The finger is connected to the shaft through the
compliance of the transmission, so a pushed finger deflects a little and springs back
when the load goes; a grasp holds without any command.

``HandMotor`` models this once per decision (60 Hz):

* the motor's drive torque is ``motor_stiffness · (target − shaft)``, limited to
  ``advancing_cap`` (the motor's torque budget);
* the shaft advances by the step that torque can make against the measured finger load
  (the compliance ``Kp · (shaft − q)`` and damping, with the finger position predicted
  one decision ahead), at most ``speed_limit`` and at most to the target;
* the step is never negative in the drive's direction: **the shaft never moves away
  from its target**, whatever the finger load.

Each physics step (480 Hz) the joint is pulled toward the shaft by LW-v0's position
drive (Kp 2500 thumb flexion, 1250 finger flexion, 600 thumb yaw; output caps 60 / 10 N·m,
integrated implicitly); the position target is the shaft, linearly interpolated over the
decision, plus a velocity feed-forward ``output_damping / Kp · shaft speed``. The five
coupled finger joints follow their actuated joints through equality constraints in
LW-v0's files.

:class:`HandAction` passes each physics step's target through the arm's torque guard
(:func:`linker_sim.arm.torque_guard`): the drive's torque ``Kp·(target − q) − kd·q̇`` stays
1 N·m (``TORQUE_GUARD_MARGIN``) within the joint's force limit, so a drive is never asked for
more than it can give. On MuJoCo 4 a drive held at its limit loses its implicit damping and
oscillates there, and the motor model, reading that as a load, would stop the shaft for good;
the guard keeps every MuJoCo version on the same physics (motor model
``geared-shaft-v1``).

The stiffness Kp and the shaft range are read from LW-v0's files (the joints' drive gains and
position limits); the motor's own constants from ``hands/o6/motor.yaml`` (``linker_sim.assets.motor``,
which also derives the passive damping each shaft sees from LW-v0's files).

Presets (:data:`PRESETS`, shaft targets in rad) are the hand poses used by the playground.

Copied from linker-lab v1.0.0:src/linker_lab/hand.py: :data:`SHAFTS`, :data:`PRESETS`,
:func:`thumb_oppose`, :class:`HandMotor` and :class:`HandActionCfg`'s ``normalized``;
change them there first (``tools/check_copies.py`` compares them). New here:
:class:`HandActionCfg` is a plain description, and :class:`HandAction` reads and writes
MuJoCo's arrays instead of an Isaac Lab articulation; its steps are linker-lab's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from linker_sim import physics
from linker_sim.arm import TORQUE_GUARD_MARGIN, _Drives, _guard_kernel

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["HandAction", "HandActionCfg", "SHAFTS", "PRESETS", "THUMB_OPPOSE_YAW", "thumb_oppose"]

if TYPE_CHECKING:
    from linker_sim.workstation import Station

SHAFTS = ("thumb pitch", "thumb yaw", "index", "middle", "ring", "pinky")
"""The six actuated joints, in action order."""

PRESETS = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "close": (0.45, 1.2, 1.45, 1.45, 1.45, 1.45),
    "pinch": (0.37, 1.04, 0.85, 0.0, 0.0, 0.0),
}
"""Shaft targets [rad]. ``thumb_oppose`` (thumb yaw 1.04, the rest unchanged) is not a full
target; see :func:`thumb_oppose`."""

THUMB_OPPOSE_YAW = 1.04
"""Thumb yaw [rad] that turns the thumb opposite the index finger (as in ``pinch``)."""


def thumb_oppose(current: torch.Tensor) -> torch.Tensor:
    """``current`` shaft targets (N, 6) with the thumb rotated opposite the index finger."""
    result = current.clone()
    result[:, 1] = THUMB_OPPOSE_YAW
    return result


class HandMotor:
    """The motor model of :mod:`linker_sim.hand` for N hands, on any torch device.

    ``kp``, ``lower`` and ``upper`` are the joints' drive stiffness and position limits
    (6 values each); ``motor`` holds the motor's constants (``linker_sim.assets.motor``).
    Shaft state is kept in float64; the per-step position targets are float32, as the
    engine's.
    """

    def __init__(self, q, kp, lower, upper, motor: dict, steps_per_decision: int, decision_dt: float):
        self.dt = decision_dt
        self.steps = steps_per_decision
        f64 = dict(device=q.device, dtype=torch.float64)
        self.kp = torch.as_tensor(kp, **f64)
        self.lower = torch.as_tensor(lower, **f64)
        self.upper = torch.as_tensor(upper, **f64)
        self.output_damping = torch.tensor(motor["output_damping"], **f64)
        self.advancing_cap = torch.tensor(motor["advancing_cap"], **f64)
        self.viscosity = torch.tensor(motor["viscosity"], **f64)
        self.passive_damping = torch.tensor(motor["passive_damping"], **f64)
        self.motor_stiffness = motor["motor_stiffness"]
        self.speed_limit = motor["speed_limit"]
        self.shaft = torch.zeros(q.shape, **f64)
        self.previous_q = torch.zeros(q.shape, **f64)
        # fractions of the decision at the end of each physics step: 1/steps, .., 1
        self._fraction = (torch.arange(1, steps_per_decision + 1, device=q.device) / steps_per_decision)
        self._fraction = self._fraction.float()[:, None, None]
        self.targets = torch.zeros((steps_per_decision, *q.shape), device=q.device)  # this decision's
        self.step_index = 0
        self.reset(q)

    def reset(self, q, rows=slice(None)):
        """Start the selected hands with the shaft at the measured joint position."""
        q = q.double()[rows]
        shaft = torch.minimum(torch.maximum(q, self.lower), self.upper)
        self.shaft[rows] = shaft
        self.previous_q[rows] = q
        self.targets[:, rows] = shaft.float()

    def decide(self, q, target):
        """Advance every shaft once toward ``target`` (N, 6) [rad], given the measured
        positions ``q`` (N, 6); return the shaft positions after the decision."""
        q, old = q.double(), self.shaft
        target = torch.minimum(torch.maximum(target.double(), self.lower), self.upper)
        q_speed = (q - self.previous_q) / self.dt
        error = target - old
        drive = (self.motor_stiffness * error).clamp(-self.advancing_cap, self.advancing_cap)
        direction = torch.sign(drive)
        # Step the drive torque can make against the finger load (backward Euler of the
        # shaft against the compliance kp and the dampings), the finger predicted ahead.
        kp, damping = self.kp, self.output_damping
        predicted_q = q + self.dt * q_speed
        possible = (drive - kp * (old - predicted_q) + damping * q_speed) / (
            kp + (damping + self.viscosity) / self.dt
        )
        # Step of the unloaded motor: speed limit, or the viscous losses at this torque.
        free = torch.minimum(
            torch.full_like(drive, self.dt * self.speed_limit),
            self.dt * drive.abs() / (self.viscosity + self.passive_damping),
        )
        # Never against the drive direction (self-locking), never past the target.
        step = direction * torch.minimum(
            (direction * possible).clamp(min=0), torch.minimum(error.abs(), free)
        )
        shaft = old + step
        speed = step / self.dt
        native = shaft + damping / kp * speed  # target with velocity feed-forward
        # per physics step: the shaft interpolated over the decision plus the feed-forward,
        # ending exactly on the feed-forward target
        start, end, offset = old.float(), shaft.float(), (native - shaft).float()
        self.targets = start + self._fraction * (end - start) + offset
        self.targets[-1] = native.float()
        self.shaft = shaft
        self.previous_q = q
        self.step_index = 0
        return shaft

    def step_target(self):
        """The position target for the next physics step (float32, N × 6)."""
        self.step_index += 1
        return self.targets[min(self.step_index, self.steps) - 1]


class HandAction:
    """Six shaft targets per decision for one O6 hand of a station, through
    :class:`HandMotor`.

    Actions are absolute radians in :data:`SHAFTS` order, or normalized: -1 is the lower
    joint limit, +1 the upper (``cfg.normalized``). Targets outside the joint range are
    clamped into it. A non-finite target keeps that shaft's previous target.

    :class:`~linker_sim.workstation.Station` makes one per hand and calls it as linker-lab's
    action manager does (see :class:`~linker_sim.arm.ArmAction`).
    """

    def __init__(self, cfg: HandActionCfg, station: Station):
        if tuple(cfg.joint_names) != tuple(cfg.motor["shafts"]):
            raise ValueError(f"HandAction: joint_names {list(cfg.joint_names)} are not the motor's shafts {cfg.motor['shafts']}")
        self.cfg = cfg
        model, self._data = station.model, station.data
        drives = _Drives(model, cfg.asset_name, cfg.joint_names)
        self._qpos, self._dof, self._ctrl = drives.qpos, drives.dof, drives.actuator
        self._lower = torch.tensor(drives.lower, dtype=torch.float32)
        self._upper = torch.tensor(drives.upper, dtype=torch.float32)
        self._motor = HandMotor(
            self._joint_pos(),
            drives.kp.astype(np.float32),  # float32, as linker-lab's engine holds it
            self._lower,
            self._upper,
            cfg.motor,
            steps_per_decision=physics.DECIMATION,
            decision_dt=physics.DECIMATION * float(model.opt.timestep),
        )
        self._raw = torch.zeros(1, 6)
        self._target = self._motor.shaft.float().clone()
        f32 = dict(dtype=torch.float32)
        self._request = torch.zeros((1, 6), **f32)  # this physics step's motor target
        self._guarded = torch.zeros((1, 6), **f32)  # the same through the torque guard
        self._q, self._qd = torch.zeros((1, 6), **f32), torch.zeros((1, 6), **f32)  # measured
        gains = [torch.tensor(x, **f32)[None] for x in (drives.kp, drives.kd, drives.cap)]
        self._guard = self._record_guard(gains)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        """The shaft targets [rad] in effect."""
        return self._target

    @property
    def shaft(self) -> torch.Tensor:
        """Motor shaft positions [rad] (1, 6)."""
        return self._motor.shaft

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        target = actions
        if self.cfg.normalized:
            target = self._lower + (actions.clamp(-1, 1) + 1) * 0.5 * (self._upper - self._lower)
        self._target = torch.where(torch.isfinite(target), target, self._target)
        self._motor.decide(self._joint_pos(), self._target)

    def apply_actions(self):
        self._request.copy_(self._motor.step_target())
        q, qd = self._q.numpy(), self._qd.numpy()  # views: the guard reads these buffers
        q[0], qd[0] = self._data.qpos[self._qpos], self._data.qvel[self._dof]  # as float32
        self._guard.launch()  # torque_guard(self._request, self._q, self._qd, ...)
        self._data.ctrl[self._ctrl] = self._guarded[0].numpy()

    def reset(self):
        self._motor.reset(self._joint_pos())
        self._raw[:] = 0
        self._target[:] = self._motor.shaft.float()

    def _record_guard(self, gains) -> wp.Launch:
        """linker-lab's ``HandAction`` passes each physics step's target through the arm's
        :func:`~linker_sim.arm.torque_guard`; here the same kernel, recorded once as a Warp
        launch on this action's buffers (as :class:`~linker_sim.arm.ArmAction` does)."""
        wp.init()  # no-op once Warp is initialized
        joints = torch.arange(6, dtype=torch.int32)
        arrays = [wp.from_torch(x) for x in (self._request, self._q, self._qd, *gains, joints)]
        self._guard_buffers = (gains, joints)  # keep the tensors the launch points at alive
        return wp.launch(
            _guard_kernel, dim=(1, 6), inputs=[*arrays, TORQUE_GUARD_MARGIN],
            outputs=[wp.from_torch(self._guarded)], device="cpu", record_cmd=True,
        )

    def _joint_pos(self) -> torch.Tensor:
        """Measured positions (1, 6) of the actuated joints, float32 as linker-lab's engine."""
        return torch.from_numpy(self._data.qpos[self._qpos].astype(np.float32))[None]


@dataclass(kw_only=True)
class HandActionCfg:
    """:class:`HandAction`. ``linker_sim.workstation.hand_action_cfg`` fills in the joints and
    the motor."""

    asset_name: str = ""
    """The arm unit: ``"left"`` or ``"right"``."""
    joint_names: Sequence[str] = ()
    """The 6 actuated joints in :data:`SHAFTS` order (names in the unit's own file): the
    motor's ``shafts``."""
    motor: dict = field(default_factory=dict)
    """The hand's motor (``linker_sim.assets.motor(side)``): its shafts and constants."""
    normalized: bool = False
    """Actions in [-1, 1] over the joint range instead of radians."""
