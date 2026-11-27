"""The AR5 arm: how it is commanded (``ArmAction``), its servo model and its motion limits.

How the real arm is commanded. The xCore SDK sends one position per 1 ms cycle to the
servo; smoothing happens before the servo, and the caller chooses who does it:

* **SDK interpolation on (PP tracking).** The caller sends goals; the SDK plans a
  jerk-limited trajectory (Ruckig, all joints arriving together) to the latest goal
  and streams it.
* **SDK interpolation off (CSP servo).** The caller streams the path itself and is
  responsible for its continuity.

``ArmAction`` models both. Each decision (60 Hz) it receives a command and turns it into
one reference (q_ref, v_ref) per physics step (480 Hz):

``interpolation=True`` (default)
    The command is a **goal**. :func:`plan` (a trajectory generator) moves the reference
    toward the latest goal: time-optimal under the velocity, acceleration and jerk
    limits, all seven joints arriving together, replanned every decision from the
    reference's own position, velocity and acceleration.
``interpolation=False``
    The command is the **next point of the caller's own path** (``stream``):
    ``"points"`` (7 joint positions; the point's velocity is the finite difference to the
    previous point, so the path never stops at a point), ``"points_velocity"`` (positions
    and velocities, 14 values), or ``"samples"`` (the 8 per-step positions, then the 8
    per-step velocities: 112 values, used as given). Consecutive points are joined by a
    cubic Hermite curve.

Commands are joint positions [rad] (``command="joint"``) or a pose of a frame on the arm
(``command="pose"``: position [m] and quaternion x, y, z, w in the arm's base frame),
converted to joint positions once per decision by one damped-least-squares step from the
measured joint positions (as Isaac Lab's ``DifferentialIKController`` with
``ik_method="dls"``, which linker-lab uses), on MuJoCo's Jacobian of the frame.

Servo model (:func:`feed_forward` per decision, :func:`torque_guard` per physics step). LW-v0's
position drives (Kp 16000 N·m/rad, Kd 600-614 N·m·s/rad, integrated implicitly)
track ``q_ref + Kd/Kp · v_ref`` (velocity feed-forward). The real arm's torque caps (120
N·m J1-J4, 100 N·m J5-J7) are the joints' actuator force ranges in LW-v0's files; when the PD
torque would exceed ``cap - 1 N·m``, the target is moved so the torque equals that guard.
Gains and caps are read from LW-v0's files.

Limits (``limits="real"``, the default; the xCore SDK motion profile of the AR5_08, read from
``arms/ar5_08/controller.yaml``): velocity :data:`VELOCITY`, acceleration :data:`ACCELERATION`,
jerk :data:`JERK`, the joint ranges in LW-v0's files, and the wrist octagon
:data:`WRIST_OCTAGON` for (J6, J7):

=====================  =================================  ==========================
                       interpolation=True                 interpolation=False
=====================  =================================  ==========================
position               goal and reference projected       point and reference
                       into the limits                    projected into the limits
velocity, acceleration respected by the generator         per-step reference rate-limited
jerk                   respected by the generator         counted, not enforced
=====================  =================================  ==========================

Every intervention is counted in :attr:`ArmAction.limit_hits` (decisions plus physics
steps in which a limit changed the command, or the jerk was exceeded), so a policy relying
on infeasible motion is visible. ``limits=None`` turns all of this off; it needs
``interpolation=False``, as the generator *is* the limits.

Copied from linker-lab v1.0.0:src/linker_lab/arm.py: the constants, the trajectory
generator (:func:`plan` and its Warp functions), :func:`feed_forward`,
:func:`torque_guard`, :func:`hermite`, :func:`project`, :func:`rate_limit` and the option
check; change them there first (``tools/check_copies.py`` compares them). The Warp kernels
run on Warp's CPU device. New here: :class:`ArmActionCfg` is a plain description, and
:class:`ArmAction` reads and writes MuJoCo's arrays instead of an Isaac Lab articulation;
its steps are linker-lab's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import warp as wp

from linker_sim import assets, physics

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["ArmAction", "ArmActionCfg", "VELOCITY", "ACCELERATION", "JERK", "WRIST_OCTAGON", "TORQUE_GUARD_MARGIN"]

wp.config.log_level = wp.LOG_WARNING  # no banner when Warp initializes

if TYPE_CHECKING:
    from linker_sim.workstation import Station

_CONTROLLER = assets.controller("left")  # controller.yaml has one set of limits for both arms

VELOCITY = _CONTROLLER["max_velocity"]
"""Maximum joint velocity [rad/s], J1..J7 (0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49)."""
ACCELERATION = _CONTROLLER["max_acceleration"]
"""Maximum joint acceleration [rad/s²], J1..J7 (10 each)."""
JERK = _CONTROLLER["max_jerk"]
"""Maximum joint jerk [rad/s³], J1..J7 (40 each)."""
WRIST_OCTAGON = _CONTROLLER["wrist_region"]
"""The region the real controller allows for (J6, J7) [rad]: a convex octagon, vertices in
order. The J6 and J7 ranges in LW-v0's files (±60°, ±90°) are its bounding box."""
TORQUE_GUARD_MARGIN = 1.0
"""The servo model keeps the PD torque this far [N·m] below each joint's cap."""


# ------------------------------------------------------------- trajectory generator
#
# One joint, state (p, v, a), goal g at rest. The time-optimal jerk-limited motion to g
# has 7 phases of constant jerk: change the velocity to a peak vp (jerk +-J, 0, -+J;
# acceleration limited to A), cruise at vp, change the velocity back to 0. Its travel
# grows with vp, so vp is found by bisection: vp = +-V with a cruise if even the fastest
# peak does not overshoot, else the vp whose travel without cruise is exactly g - p.
# The joints are then synchronized: every joint is slowed to the slowest one's
# duration T by lowering its peak velocity (and cruising longer) until it takes T.

F64 = wp.float64
_vec7 = wp.types.vector(length=7, dtype=wp.float64)
_BISECTIONS = 50  # vp to 2^-50 of its range


@wp.struct
class _Profile:
    duration: _vec7  # [s] of the 7 phases
    jerk: _vec7  # [rad/s³] in each phase


@wp.func
def _advance(s: wp.vec3d, jerk: F64, t: F64) -> wp.vec3d:
    """State (p, v, a) after ``t`` seconds at constant ``jerk``."""
    return wp.vec3d(
        s[0] + t * (s[1] + t * (s[2] * F64(0.5) + t * jerk / F64(6.0))),
        s[1] + t * (s[2] + t * jerk * F64(0.5)),
        s[2] + t * jerk,
    )


@wp.func
def _velocity_change(v0: F64, a0: F64, vt: F64, amax: F64, jmax: F64) -> wp.vec4d:
    """Fastest change from (v0, a0) to velocity vt with zero acceleration.

    Jerk ``s·jmax`` for t1, 0 for t2, ``-s·jmax`` for t3; returns (t1, t2, t3, s).
    """
    zero = F64(0.0)
    s = F64(1.0)
    if vt < v0 + a0 * wp.abs(a0) / (F64(2.0) * jmax):  # velocity once a0 is ramped to 0
        s = F64(-1.0)
    a0s = s * a0
    dv = s * (vt - v0)
    peak = wp.sqrt(wp.max(jmax * dv + F64(0.5) * a0s * a0s, zero))
    t2 = zero
    if peak > amax:
        peak = amax
        t2 = wp.max((dv - (F64(2.0) * amax * amax - a0s * a0s) / (F64(2.0) * jmax)) / amax, zero)
    return wp.vec4d(wp.max((peak - a0s) / jmax, zero), t2, peak / jmax, s)


@wp.func
def _profile(v0: F64, a0: F64, vp: F64, cruise: F64, amax: F64, jmax: F64) -> _Profile:
    up = _velocity_change(v0, a0, vp, amax, jmax)
    down = _velocity_change(vp, F64(0.0), F64(0.0), amax, jmax)
    out = _Profile()
    out.duration = _vec7(up[0], up[1], up[2], cruise, down[0], down[1], down[2])
    zero = F64(0.0)
    out.jerk = _vec7(up[3] * jmax, zero, -up[3] * jmax, zero, down[3] * jmax, zero, -down[3] * jmax)
    return out


@wp.func
def _state_at(s0: wp.vec3d, profile: _Profile, t: F64) -> wp.vec3d:
    s = s0
    remaining = t
    for i in range(7):
        s = _advance(s, profile.jerk[i], wp.min(profile.duration[i], wp.max(remaining, F64(0.0))))
        remaining = remaining - profile.duration[i]
    return s


@wp.func
def _duration(profile: _Profile) -> F64:
    total = F64(0.0)
    for i in range(7):
        total = total + profile.duration[i]
    return total


@wp.func
def _travel(v0: F64, a0: F64, vp: F64, amax: F64, jmax: F64) -> wp.vec2d:
    """Distance and time to change the velocity to vp and back to rest, without cruise."""
    profile = _profile(v0, a0, vp, F64(0.0), amax, jmax)
    took = _duration(profile)
    return wp.vec2d(_state_at(wp.vec3d(F64(0.0), v0, a0), profile, took)[0], took)


@wp.func
def _fastest(e: F64, v0: F64, a0: F64, vmax: F64, amax: F64, jmax: F64) -> wp.vec3d:
    """Peak velocity, cruise time and duration of the time-optimal motion over distance e."""
    zero = F64(0.0)
    result = wp.vec3d(zero, zero, zero)  # at rest at the goal: nothing to do
    if e != zero or v0 != zero or a0 != zero:
        up = _travel(v0, a0, vmax, amax, jmax)
        down = _travel(v0, a0, -vmax, amax, jmax)
        if e >= up[0]:
            result = wp.vec3d(vmax, (e - up[0]) / vmax, up[1] + (e - up[0]) / vmax)
        elif e <= down[0]:
            result = wp.vec3d(-vmax, (e - down[0]) / (-vmax), down[1] + (e - down[0]) / (-vmax))
        else:
            lo = -vmax
            hi = vmax
            for _ in range(_BISECTIONS):
                mid = F64(0.5) * (lo + hi)
                if _travel(v0, a0, mid, amax, jmax)[0] < e:
                    lo = mid
                else:
                    hi = mid
            vp = F64(0.5) * (lo + hi)
            result = wp.vec3d(vp, zero, _travel(v0, a0, vp, amax, jmax)[1])
    return result


@wp.func
def _within(T: F64, e: F64, v0: F64, a0: F64, fastest_peak: F64, amax: F64, jmax: F64) -> wp.vec2d:
    """Peak velocity and cruise time of a motion over distance e that takes T (at least as
    long as the fastest one): the same shape with a lower peak velocity."""
    zero = F64(0.0)
    stop = _travel(v0, a0, zero, amax, jmax)
    rest = e - stop[0]  # distance left after stopping
    result = wp.vec2d(zero, wp.max(T - stop[1], zero))  # stopping arrives: stop, then wait
    if wp.abs(rest) > F64(1.0e-12):
        sign = wp.sign(rest)
        lo = zero
        hi = wp.abs(fastest_peak)
        for _ in range(_BISECTIONS):
            mid = F64(0.5) * (lo + hi)
            travel = _travel(v0, a0, sign * mid, amax, jmax)
            if travel[1] + (e - travel[0]) / (sign * mid) > T:  # too slow
                lo = mid
            else:
                hi = mid
        vp = sign * hi
        result = wp.vec2d(vp, wp.max((e - _travel(v0, a0, vp, amax, jmax)[0]) / vp, zero))
    return result


@wp.kernel
def _fastest_kernel(
    q: wp.array2d(dtype=F64),
    v: wp.array2d(dtype=F64),
    a: wp.array2d(dtype=F64),
    goal: wp.array2d(dtype=F64),
    vmax: wp.array(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    out: wp.array3d(dtype=F64),
):
    n, j = wp.tid()
    f = _fastest(goal[n, j] - q[n, j], v[n, j], a[n, j], vmax[j], amax[j], jmax[j])
    out[n, j, 0] = f[0]
    out[n, j, 1] = f[1]
    out[n, j, 2] = f[2]


@wp.kernel
def _sample_kernel(
    q: wp.array2d(dtype=F64),
    v: wp.array2d(dtype=F64),
    a: wp.array2d(dtype=F64),
    goal: wp.array2d(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    fastest: wp.array3d(dtype=F64),
    slowest: wp.array(dtype=F64),
    dt: F64,
    out_q: wp.array3d(dtype=F64),
    out_v: wp.array3d(dtype=F64),
    out_a: wp.array3d(dtype=F64),
):
    n, j = wp.tid()
    f = wp.vec2d(fastest[n, j, 0], fastest[n, j, 1])
    if fastest[n, j, 2] < slowest[n] - F64(1.0e-9):  # slow down to arrive with the slowest joint
        f = _within(slowest[n], goal[n, j] - q[n, j], v[n, j], a[n, j], f[0], amax[j], jmax[j])
    profile = _profile(v[n, j], a[n, j], f[0], f[1], amax[j], jmax[j])
    total = _duration(profile)
    s0 = wp.vec3d(q[n, j], v[n, j], a[n, j])
    for k in range(out_q.shape[1]):
        t = F64(k + 1) * dt
        s = wp.vec3d(goal[n, j], F64(0.0), F64(0.0))  # arrived
        if t < total:
            s = _state_at(s0, profile, t)
        out_q[n, k, j] = s[0]
        out_v[n, k, j] = s[1]
        out_a[n, k, j] = s[2]


def plan(q, v, a, goal, vmax, amax, jmax, dt: float, steps: int):
    """Sample the trajectory from state (q, v, a) to ``goal`` at t = dt, 2·dt, .., steps·dt.

    All inputs are float64 torch tensors on one device: states and goal (N, 7), limits (7,).
    The state must be within the limits and able to stop within them (any state on a
    previous plan is). Returns positions, velocities and accelerations (N, steps, 7):
    time-optimal under the limits with every joint arriving at its goal together, at
    rest from then on.
    """
    wp.init()  # no-op once Warp is initialized
    n = q.shape[0]
    device = wp.device_from_torch(q.device)
    stream = wp.stream_from_torch(q.device) if device.is_cuda else None
    state = [wp.from_torch(x.contiguous(), dtype=wp.float64) for x in (q, v, a, goal)]
    limits = [wp.from_torch(x.contiguous(), dtype=wp.float64) for x in (vmax, amax, jmax)]
    fastest = torch.empty((n, 7, 3), device=q.device, dtype=torch.float64)  # peak, cruise, duration
    wp.launch(_fastest_kernel, dim=(n, 7), inputs=[*state, *limits], outputs=[wp.from_torch(fastest)],
              device=device, stream=stream)
    slowest = fastest[:, :, 2].amax(1).contiguous()
    out = [torch.empty((n, steps, 7), device=q.device, dtype=torch.float64) for _ in range(3)]
    wp.launch(
        _sample_kernel,
        dim=(n, 7),
        inputs=[*state, *limits[1:], wp.from_torch(fastest), wp.from_torch(slowest), float(dt)],
        outputs=[wp.from_torch(x) for x in out],
        device=device,
        stream=stream,
    )
    return tuple(out)


# ------------------------------------------------------------- servo, stream, limits


def feed_forward(q_ref, v_ref, kp, kd):
    """The servo's position target before the torque guard: ``q_ref + kd/kp · v_ref``.

    Computed in float64 and rounded to the engine's float32; inputs broadcast.
    """
    return (q_ref.double() + kd.double() * v_ref.double() / kp.double()).float()


@wp.kernel
def _guard_kernel(
    requested: wp.array2d(dtype=wp.float32),
    q: wp.array2d(dtype=wp.float32),
    qd: wp.array2d(dtype=wp.float32),
    kp: wp.array2d(dtype=wp.float32),
    kd: wp.array2d(dtype=wp.float32),
    cap: wp.array2d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    margin: wp.float32,
    out: wp.array2d(dtype=wp.float32),
):
    n, i = wp.tid()
    j = joints[i]
    guard = cap[n, j] - margin
    target = requested[n, i]
    torque = kp[n, j] * (target - q[n, j]) - kd[n, j] * qd[n, j]
    if wp.abs(torque) > guard:  # the target that gives exactly the guard torque
        target = q[n, j] + (kd[n, j] * qd[n, j] + wp.clamp(torque, -guard, guard)) / kp[n, j]
    out[n, i] = target


def torque_guard(requested, q, qd, kp, kd, cap, joints, out=None, margin: float = TORQUE_GUARD_MARGIN):
    """Position targets (N, K) for one physics step: ``requested`` (N, K), unless the PD torque
    ``kp·(requested − q) − kd·qd`` would exceed ``cap − margin``; then the target that gives
    exactly that torque.

    ``q``, ``qd``, ``kp``, ``kd``, ``cap`` are the articulation's float32 joint arrays (N, J)
    (Warp arrays or torch tensors); ``joints`` (K, int32) selects the driven columns: the
    arm's seven, or a hand's six (``linker_sim.hand.HandAction``).
    """
    wp.init()  # no-op once Warp is initialized
    out = torch.empty_like(requested) if out is None else out
    device = wp.device_from_torch(requested.device)
    arrays = [x if isinstance(x, wp.array) else wp.from_torch(x) for x in (requested, q, qd, kp, kd, cap, joints)]
    wp.launch(_guard_kernel, dim=requested.shape, inputs=[*arrays, float(margin)], outputs=[wp.from_torch(out)],
              device=device, stream=wp.stream_from_torch(requested.device) if device.is_cuda else None)
    return out


def hermite(q0, q1, v0, v1, u, duration: float):
    """Cubic Hermite curve from (q0, v0) to (q1, v1) over ``duration``, at fractions ``u``.

    ``q0``..``v1`` are (N, 7), ``u`` is (S, 1); returns positions and velocities (N, S, 7).
    """
    q0, q1, v0, v1 = (x[:, None, :] for x in (q0, q1, v0, v1))
    q = (
        (2 * u**3 - 3 * u**2 + 1) * q0
        + (u**3 - 2 * u**2 + u) * duration * v0
        + (-2 * u**3 + 3 * u**2) * q1
        + (u**3 - u**2) * duration * v1
    )
    v = (
        (6 * u**2 - 6 * u) * q0
        + (3 * u**2 - 4 * u + 1) * duration * v0
        + (-6 * u**2 + 6 * u) * q1
        + (3 * u**2 - 2 * u) * duration * v1
    ) / duration
    return q, v


@wp.kernel
def _project_kernel(
    q: wp.array2d(dtype=F64),
    lower: wp.array(dtype=F64),
    upper: wp.array(dtype=F64),
    corners: wp.array2d(dtype=F64),
    hit: wp.array(dtype=wp.int32),
):
    m = wp.tid()
    changed = wp.int32(0)
    for j in range(7):
        clamped = wp.clamp(q[m, j], lower[j], upper[j])
        if clamped != q[m, j]:
            changed = 1
            q[m, j] = clamped
    # wrist (J6, J7): inside the octagon, or to the nearest point on its boundary
    p = wp.vec2d(q[m, 5], q[m, 6])
    inside = True
    nearest = p
    best = F64(1.0e30)
    for i in range(8):
        a = wp.vec2d(corners[i, 0], corners[i, 1])
        edge = wp.vec2d(corners[(i + 1) % 8, 0], corners[(i + 1) % 8, 1]) - a
        r = p - a
        if edge[0] * r[1] - edge[1] * r[0] > F64(0.0):  # left of a clockwise edge: outside
            inside = False
        c = a + wp.clamp(wp.dot(r, edge) / wp.dot(edge, edge), F64(0.0), F64(1.0)) * edge
        if wp.length_sq(p - c) < best:
            best = wp.length_sq(p - c)
            nearest = c
    if not inside:
        changed = 1
        q[m, 5] = nearest[0]
        q[m, 6] = nearest[1]
    hit[m] = changed


def project(q, lower, upper):
    """Joint positions (..., 7) moved into the joint ranges ``lower``, ``upper`` (7,) and the
    wrist octagon. Returns the positions (float64) and where they changed (bool, q.shape[:-1])."""
    wp.init()  # no-op once Warp is initialized
    out = q.double().reshape(-1, 7).clone()
    hit = torch.empty(out.shape[0], device=q.device, dtype=torch.int32)
    corners = torch.tensor(WRIST_OCTAGON, device=q.device, dtype=torch.float64)
    device = wp.device_from_torch(q.device)
    wp.launch(_project_kernel, dim=out.shape[0],
              inputs=[wp.from_torch(out)] + [wp.from_torch(x.double().contiguous()) for x in (lower, upper, corners)],
              outputs=[wp.from_torch(hit)], device=device,
              stream=wp.stream_from_torch(q.device) if device.is_cuda else None)
    return out.reshape(q.shape), hit.reshape(q.shape[:-1]).bool()


@wp.kernel
def _rate_limit_kernel(
    q: wp.array3d(dtype=F64),
    v: wp.array3d(dtype=F64),
    last: wp.array3d(dtype=F64),
    vmax: wp.array(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    dt: F64,
    hit: wp.array2d(dtype=wp.int32),
):
    n, j = wp.tid()
    last_q = last[n, j, 0]
    last_v = last[n, j, 1]  # finite-difference velocity of the last step
    last_a = last[n, j, 2]
    changed = wp.int32(0)
    for k in range(q.shape[1]):
        wanted = (q[n, k, j] - last_q) / dt
        vel = wp.clamp(wp.clamp(wanted, -vmax[j], vmax[j]), last_v - amax[j] * dt, last_v + amax[j] * dt)
        if vel != wanted:
            changed = 1
            q[n, k, j] = last_q + vel * dt
            v[n, k, j] = vel
        acc = (vel - last_v) / dt
        if wp.abs(acc - last_a) > jmax[j] * dt * F64(1.000000001):  # jerk: counted only
            changed = 1
        last_q = q[n, k, j]
        last_v = vel
        last_a = acc
    last[n, j, 0] = last_q
    last[n, j, 1] = last_v
    last[n, j, 2] = last_a
    hit[n, j] = changed


def rate_limit(q, v, last, vmax, amax, jmax, dt: float):
    """The servo-mode limits on per-step references q, v (N, S, 7), in place: each step's
    finite-difference velocity limited to ``vmax`` and its change to ``amax``·dt; the jerk
    is only counted. ``last`` (N, 7, 3) holds the last step's position, finite-difference
    velocity and acceleration and is updated. Returns where a limit acted (bool, N)."""
    wp.init()  # no-op once Warp is initialized
    hit = torch.empty(q.shape[0], 7, device=q.device, dtype=torch.int32)
    device = wp.device_from_torch(q.device)
    wp.launch(_rate_limit_kernel, dim=(q.shape[0], 7),
              inputs=[wp.from_torch(x) for x in (q, v, last, vmax, amax, jmax)] + [float(dt)],
              outputs=[wp.from_torch(hit)], device=device,
              stream=wp.stream_from_torch(q.device) if device.is_cuda else None)
    return hit.any(-1)


# ------------------------------------------------------------- action


class ArmAction:
    """Joint or pose commands for one AR5 arm of a station, through the trajectory generator
    or the Hermite stream, into the servo model. See :mod:`linker_sim.arm`.

    :class:`~linker_sim.workstation.Station` makes one per arm and calls it as linker-lab's
    action manager does: :meth:`process_actions` once per decision, :meth:`apply_actions`
    before each physics step, :meth:`reset` after the station's reset. Tensors have one row
    (one station) and are on the CPU.
    """

    def __init__(self, cfg: ArmActionCfg, station: Station):
        _check(cfg)
        if len(cfg.joint_names) != 7:
            raise ValueError(f"ArmAction needs the 7 arm joints, got {list(cfg.joint_names)}")
        model, self._data = station.model, station.data
        self.cfg = cfg
        drives = _Drives(model, cfg.asset_name, cfg.joint_names)
        self._qpos, self._dof, self._ctrl = drives.qpos, drives.dof, drives.actuator
        f32 = dict(dtype=torch.float32)
        self._kp = torch.tensor(drives.kp, **f32)[None]
        self._kd = torch.tensor(drives.kd, **f32)[None]
        self._cap = torch.tensor(drives.cap, **f32)[None]
        self._steps = physics.DECIMATION
        self._dt = float(model.opt.timestep)
        self._decision_dt = self._steps * self._dt
        f64 = dict(dtype=torch.float64)
        self._lower = torch.tensor(drives.lower, **f64)
        self._upper = torch.tensor(drives.upper, **f64)
        self._vmax = torch.tensor(VELOCITY, **f64)
        self._amax = torch.tensor(ACCELERATION, **f64)
        self._jmax = torch.tensor(JERK, **f64)
        self._u = torch.arange(1, self._steps + 1, **f64)[:, None] / self._steps

        n, s = 1, self._steps
        self._ref_q = torch.zeros((n, s, 7), **f64)  # per-step references of this decision
        self._ref_v = torch.zeros((n, s, 7), **f64)
        self._requested = torch.zeros((s, n, 7), **f32)  # their servo targets
        self._target = torch.zeros((n, 7), **f32)
        self._joints = torch.arange(7, dtype=torch.int32)  # the guard's columns: all seven
        self._q, self._qd = torch.zeros((n, 7), **f32), torch.zeros((n, 7), **f32)  # measured
        self._guard = self._record_guard()
        self._step = 0
        self._plan = [torch.zeros((n, 7), **f64) for _ in range(3)]  # generator state q, v, a
        self._point = [torch.zeros((n, 7), **f64) for _ in range(2)]  # last stream point q, v
        self._last = torch.zeros((n, 7, 3), **f64)  # last reference: q, velocity, acceleration
        self.limit_hits = torch.zeros(n, dtype=torch.long)
        """Decisions plus physics steps since the last reset in which a limit changed the
        command, or the jerk limit was exceeded (one value: this arm)."""
        self._raw = torch.zeros((n, self.action_dim), **f32)

        if cfg.command == "pose":
            self._model = model
            self._body = model.body(f"{cfg.asset_name}/{cfg.body_name}").id
            self._base = model.body_rootid[self._body]  # the arm unit's base, welded to the world
            self._offset = np.array(cfg.body_offset, dtype=np.float64)
        self.reset()

    # -- the action interface (linker-lab's ActionTerm)

    @property
    def action_dim(self) -> int:
        if self.cfg.stream == "points_velocity":
            return 14
        if self.cfg.stream == "samples":
            return 2 * self._steps * 7
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        """The last command; non-finite values are replaced by the previous command's."""
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The reference (q_ref, v_ref) of the current physics step, (1, 7) each."""
        k = min(max(self._step - 1, 0), self._steps - 1)
        return self._ref_q[:, k], self._ref_v[:, k]

    def process_actions(self, actions: torch.Tensor):
        self._raw = torch.where(torch.isfinite(actions), actions, self._raw)
        limits = self.cfg.limits is not None
        hits = torch.zeros_like(self.limit_hits, dtype=torch.bool)

        if self.cfg.stream == "samples":
            s = self._steps
            ref_q = self._raw[:, : s * 7].view(-1, s, 7).double()
            ref_v = self._raw[:, s * 7 :].view(-1, s, 7).double()
        else:
            point = self._joint_command(self._raw)
            if limits:
                point, changed = project(point, self._lower, self._upper)
                hits |= changed
            if self.cfg.interpolation:
                ref_q, ref_v, ref_a = plan(
                    *self._plan, point, self._vmax, self._amax, self._jmax, self._dt, self._steps
                )
                self._plan = [x[:, -1].clone() for x in (ref_q, ref_v, ref_a)]
            else:
                q0, v0 = self._point
                if self.cfg.stream == "points_velocity":
                    v1 = self._raw[:, 7:].double()
                else:  # the finite difference, so the path does not stop at the point
                    v1 = (point - q0) / self._decision_dt
                ref_q, ref_v = hermite(q0, point, v0, v1, self._u, self._decision_dt)
                self._point = [point, v1]

        if limits:
            ref_q, changed = project(ref_q, self._lower, self._upper)
            hits |= changed.any(-1)
            if not self.cfg.interpolation:
                ref_v = ref_v.contiguous()
                hits |= rate_limit(ref_q, ref_v, self._last, self._vmax, self._amax, self._jmax, self._dt)
        self._ref_q, self._ref_v = ref_q, ref_v
        self._requested.copy_(feed_forward(ref_q, ref_v, self._kp[:, None], self._kd[:, None]).transpose(0, 1))
        self._step = 0
        self.limit_hits += hits

    def apply_actions(self):
        k = min(self._step, self._steps - 1)
        self._step += 1
        q, qd = self._q.numpy(), self._qd.numpy()  # views: the guard reads these buffers
        q[0], qd[0] = self._data.qpos[self._qpos], self._data.qvel[self._dof]  # as float32
        self._guard[k].launch()  # torque_guard(self._requested[k], self._q, self._qd, ...)
        self._data.ctrl[self._ctrl] = self._target[0].numpy()

    def reset(self):
        q = torch.from_numpy(self._data.qpos[self._qpos])[None]  # float64
        for state in (self._plan, self._point):
            state[0][:] = q
            for derivative in state[1:]:
                derivative[:] = 0
        self._last[:] = 0
        self._last[:, :, 0] = q
        self._ref_q[:] = q[:, None, :]
        self._ref_v[:] = 0
        self._requested[:] = q.float()
        self._step = 0
        self.limit_hits[:] = 0
        if self.cfg.command == "pose":
            pos, quat = self._frame_pose()
            self._raw[:] = torch.from_numpy(np.concatenate((pos, quat)))[None].float()
        elif self.cfg.stream == "samples":
            self._raw[:] = torch.cat(
                (q[:, None, :].expand(-1, self._steps, 7).flatten(1), torch.zeros_like(q).repeat(1, self._steps)), -1
            ).float()
        else:
            self._raw[:] = 0
            self._raw[:, :7] = q.float()

    # -- MuJoCo

    def _record_guard(self) -> list:
        """:func:`torque_guard` of each physics step of a decision, recorded once as a Warp
        launch on this action's buffers: the same kernel and arguments, without wrapping
        seven tensors anew every physics step (which made the guard most of the station's
        step time). Outputs into ``self._target``; float32 measured joint state, as
        linker-lab's engine holds it."""
        wp.init()  # no-op once Warp is initialized
        arrays = [wp.from_torch(x) for x in (self._q, self._qd, self._kp, self._kd, self._cap, self._joints)]
        return [
            wp.launch(
                _guard_kernel, dim=(1, 7), inputs=[wp.from_torch(self._requested[k]), *arrays, TORQUE_GUARD_MARGIN],
                outputs=[wp.from_torch(self._target)], device="cpu", record_cmd=True,
            )
            for k in range(self._steps)
        ]

    def _joint_command(self, raw):
        """Joint positions (1, 7, float64) the command asks for."""
        if self.cfg.command == "joint":
            return raw[:, :7].double()
        goal = raw[0].double().numpy()
        q = self._data.qpos[self._qpos]
        pos, quat = self._frame_pose()
        return torch.from_numpy(q + _dls_step(pos, quat, goal[:3], goal[3:], self._frame_jacobian(), self.cfg.ik_damping))[None]

    def _frame_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Position (3,) and quaternion (4,) x, y, z, w of the commanded frame in the arm's
        base frame."""
        data, body, base = self._data, self._body, self._base
        base_r = data.xmat[base].reshape(3, 3)
        body_r = data.xmat[body].reshape(3, 3)
        pos = base_r.T @ (data.xpos[body] + body_r @ self._offset - data.xpos[base])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (base_r.T @ body_r).flatten())
        return pos, np.array([*quat[1:], quat[0]])

    def _frame_jacobian(self) -> np.ndarray:
        """Geometric Jacobian (6, 7) of the commanded frame in the arm's base frame: linear
        velocity of the frame's origin, then angular velocity."""
        data, body = self._data, self._body
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ self._offset
        linear, angular = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
        mujoco.mj_jac(self._model, data, linear, angular, point, body)
        base_r = data.xmat[self._base].reshape(3, 3)
        return np.vstack((base_r.T @ linear[:, self._dof], base_r.T @ angular[:, self._dof]))


def _dls_step(pos, quat, goal_pos, goal_quat, jacobian, damping: float) -> np.ndarray:
    """Joint step (7,) of damped least squares toward a goal pose, quaternions x, y, z, w:
    ``J^T (J J^T + damping^2 I)^-1 e``, with ``e`` the position error and the rotation
    vector from the frame's orientation to the goal's. The goal quaternion is normalized;
    if it cannot be (all zero), the orientation is held. As Isaac Lab's
    ``DifferentialIKController(command_type="pose", ik_method="dls")``."""
    norm = np.linalg.norm(goal_quat)
    goal_quat = goal_quat / norm if norm > 0 and np.isfinite(norm) else quat
    wxyz = lambda q: np.array([q[3], q[0], q[1], q[2]])  # noqa: E731
    inverse, difference, rotation = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inverse, wxyz(quat))
    mujoco.mju_mulQuat(difference, wxyz(goal_quat), inverse)
    mujoco.mju_quat2Vel(rotation, difference, 1.0)  # the shorter way round
    error = np.concatenate((goal_pos - pos, rotation))
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)


class _Drives:
    """The position drives of some joints of one arm unit, as LW-v0's files define them."""

    def __init__(self, model: mujoco.MjModel, side: str, names: Sequence[str]):
        joints = [model.joint(f"{side}/{name}").id for name in names]
        actuator_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        missing = [name for name, j in zip(names, joints) if j not in actuator_of]
        if missing:
            raise ValueError(f"no drive on {side} joints {missing}")
        self.actuator = np.array([actuator_of[j] for j in joints])
        self.qpos = model.jnt_qposadr[joints]
        self.dof = model.jnt_dofadr[joints]
        a = self.actuator
        self.kp = model.actuator_gainprm[a, 0].copy()
        self.kd = -model.actuator_biasprm[a, 2]
        self.cap = model.jnt_actfrcrange[joints, 1].copy()
        self.lower, self.upper = model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy()
        # every drive is a position servo: force = kp (ctrl - q) - kd qd, capped at the joint
        fixed, affine = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        if not (
            (model.actuator_gaintype[a] == fixed).all()
            and (model.actuator_biastype[a] == affine).all()
            and np.array_equal(model.actuator_biasprm[a, 1], -self.kp)
            and (model.jnt_actfrclimited[joints]).all()
            and np.array_equal(model.jnt_actfrcrange[joints, 0], -self.cap)
            and (model.jnt_limited[joints]).all()
        ):
            raise ValueError(f"the {side} drives of {list(names)} are not capped position servos")


@dataclass(kw_only=True)
class ArmActionCfg:
    """:class:`ArmAction`. ``linker_sim.workstation.arm_action_cfg`` fills in the names."""

    asset_name: str
    """The arm unit: ``"left"`` or ``"right"``."""
    joint_names: Sequence[str]
    """The arm joints J1..J7 (names in the unit's own file, without the side prefix)."""
    command: str = "joint"
    """``"joint"``: 7 joint positions [rad]. ``"pose"``: position [m] and quaternion (x, y, z,
    w) of the frame ``body_name`` + ``body_offset`` in the arm's base frame."""
    interpolation: bool = True
    """True: the command is a goal for the trajectory generator (SDK PP tracking). False: it
    is the next point of the caller's own path (SDK CSP servo)."""
    stream: str = "points"
    """With ``interpolation=False``: ``"points"``, ``"points_velocity"`` or ``"samples"``."""
    limits: str | None = "real"
    """``"real"``: the real controller's limits. None: no limits (``interpolation=False`` only)."""
    body_name: str | None = None
    """Pose commands: the body whose frame is commanded."""
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Pose commands: the commanded point in that body's frame [m]."""
    ik_damping: float = 0.01
    """Damping of the least-squares IK step."""

def _check(cfg: ArmActionCfg):
    if cfg.command not in ("joint", "pose"):
        raise ValueError(f"command must be 'joint' or 'pose', not {cfg.command!r}")
    if cfg.stream not in ("points", "points_velocity", "samples"):
        raise ValueError(f"stream must be 'points', 'points_velocity' or 'samples', not {cfg.stream!r}")
    if cfg.limits not in ("real", None):
        raise ValueError(f"limits must be 'real' or None, not {cfg.limits!r}")
    if cfg.interpolation and cfg.stream != "points":
        raise ValueError("with interpolation=True the command is a goal: stream must be 'points'")
    if cfg.interpolation and cfg.limits is None:
        raise ValueError("interpolation=True needs limits='real' (the generator moves within them)")
    if cfg.command == "pose" and cfg.stream != "points":
        raise ValueError("pose commands are points; velocities and samples need joint commands")
    if cfg.command == "pose" and cfg.body_name is None:
        raise ValueError("pose commands need body_name")

