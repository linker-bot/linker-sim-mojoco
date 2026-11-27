"""Grasp YCB objects with the left hand (gallery entry ``ycb-grasp``).

``linker-sim gallery run ycb-grasp`` opens one LW-v0 station in a viewer (``--viewer
studio|viser``). Pick an object in the panel and press **Drop**: it is placed at a random
spot in the region of the table the policy was trained on (the left half: ``demo/region`` in
``policy.npz``), turned at random about the vertical, in one of its resting poses chosen at
random. Once it has come to rest, press **Grasp**: the policy
looks at it once (512 points on its surface, its pose, the robot's state) and corrects a
recorded grasp demonstration (move and turn the palm's path, offset the six finger
shafts); the grasp then runs open loop, the hand lifts the object 15 cm and holds it. It
counts as grasped if, for 2 s, the object is at least 10 cm above where it rested and
touches nothing but the left hand. **Reset** returns the robot home and clears the table.

The episode is the one the policy was trained with. Change it and the policy no longer
sees or does what it learned.

Robot: LW-v0 as shipped. The left arm streams its own path through
``ArmAction(interpolation=False)`` (60 Hz points within the real controller's limits: the
xCore SDK's servo mode, as a planned path is deployed on the real arm); the left hand takes
shaft targets (``HandAction``); the right unit holds its home pose.

Episode (ticks at 60 Hz; the demonstration's phases are in ``policy.npz``):

1. Drop: robot at home; the object in a resting pose, turned by a random yaw, the centre of
   its bounds uniform over the placement region (x0..x1, y0..y1 in the LW-v0 frame; for this
   policy x 0.2-0.62 m, y 0-0.56 m), released 0.2 mm above the table; 60 ticks to settle. It has come to rest if it is still on the
   table, within 8 cm of where it was placed, and quiet for the last 30 ticks (moved < 1 cm/s
   and turned < 0.1 rad/s, from its pose change over 0.1 s); otherwise it is placed again,
   at most twice.
2. Observation (1,595 values): left arm and hand joint positions normalized to [-1, 1] by
   their ranges (13), joint velocities (13, rad/s), hand motor shafts normalized (6), object
   position relative to the demonstration object / 0.1 m (3), object quaternion x, y, z, w
   (4), object velocity (6), palm point pose in the object frame (3 / 0.5 m, 4), the
   demonstration's grasp pose (palm at the start of closing, moved onto the object) in the
   object frame (3 / 0.5 m, 4), then the 512 surface points turned by the object's
   orientation / 0.25 m. The action (12 values in [-1, 1]): palm translation up to 5, 5,
   8 cm (x, y, z), rotation about the demonstration object up to 90 deg per Euler xyz
   angle, six shaft offsets up to 1 rad.
3. Plan: the demonstration's palm path is moved onto the object (translated by the object's
   bounds centre minus the demonstration object's position, plus the translation; turned
   by the rotation), both blended in over the approach; each tick's palm pose solved by IK
   within the real limits. Shaft offsets are blended in while the hand closes. A plan that
   misses a pose by more than 0.5 mm / 0.01 rad or jumps more than 0.05 rad between ticks
   is not executed.
4. Execute; then lift the palm point 15 cm straight up with J1-J5 (the wrist held, so it
   stays inside its octagon), then hold for 2 s.

All objects are objects of the one station, since MuJoCo fixes its bodies when the
model is built. The ones not on the table are parked below the floor, 0.6 m apart. They
fall together, side by side, with nothing below them; one of them is put back each
decision, so none falls more than about 2 m and none ever touches anything.

Ported from linker-lab v1.0.0:policy-gallery/ycb-grasp/play.py, with the same artifacts.
Copied unchanged (``tools/check_copies.py`` compares them): the constants but
:data:`PARKING`, the artifact loaders, the policy, the arm kinematics, ``parking``,
``merged``, and the episode's logic and plans (``GraspScene.press``, ``_drop``, ``_grasp``,
``_settle_check``, ``_grasp_targets``, ``plan``, ``plan_lift``); change them there first.
New here: the scene is a :class:`~linker_sim.workstation.Station`; the state is read from
MuJoCo; "touches nothing but the hand" is read from MuJoCo's contacts instead of a contact
sensor; the panel is a ``linker_sim.viewer.Panel``. Isaac Lab's quaternion functions, which
the observation and the plans use, are copied from Isaac Lab (see ``NOTICE``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from torch import nn

from linker_sim import assets, objects, workstation
from linker_sim import viewer as viewers
from linker_sim.arm import project
from linker_sim.viewer import Panel

STATE_DIM = 59
POINTS = 512
ACTION_SCALE = (0.05, 0.05, 0.08, math.pi / 2, math.pi / 2, math.pi / 2, 1, 1, 1, 1, 1, 1)
DROP_CLEARANCE = 0.0002
SETTLE_TICKS = 60
QUIET_TICKS = 30
WINDOW = 6  # ticks over which the settling motion is measured (0.1 s)
RETRIES = 2
LIFT_HEIGHT = 0.15
RISE = 0.10
IK_ITERATIONS = 50
PLAN_POSITION_TOLERANCE = 5e-4
PLAN_ROTATION_TOLERANCE = 0.01
PLAN_MAX_STEP = 0.05
CONTACT_TOLERANCE = 0.01
JOINT_LIMIT_TOLERANCE = 0.05

PARKING = {"origin": (-1.5, -1.5, -2.0), "spacing": 0.6, "columns": 7}
"""Where the objects not on the table wait [m], station frame: a grid below the floor (its
top at ``workstation.FLOOR_TOP``, -1.267 m), so no view from above the floor shows them,
wider apart than the largest object (0.28 m). linker-lab parks them 20 m down; here that
would raise MuJoCo's model extent from 8.5 to 23 m, and the viewers scale their near
clipping plane and default view with it."""
CAMERA = {"eye": (1.05, 0.75, -0.1), "lookat": (0.4, 0.28, -0.45)}
"""Initial viewer camera [m], environment frame: in front of the table, beside the left hand."""

# ---------------------------------------------------------------------------- artifacts


def load_objects(path) -> list[dict]:
    """The objects, each with its collision pieces, mass properties, surface points,
    bounds, and its resting poses (quaternion x, y, z, w; height of the origin above the
    table top). ``objects.npz`` has one entry per resting pose, named ``<id>@<pose>``."""
    result: dict[str, dict] = {}
    with np.load(path) as d:
        for i, entry in enumerate(d["names"]):
            name = str(entry).split("@")[0]
            pose = (d["rest_quat"][i], float(d["rest_height"][i]))
            if name in result:
                result[name]["poses"].append(pose)
                continue
            parts = [(d[f"v{i}_{k}"], d[f"f{i}_{k}"]) for k in range(int(d["piece_counts"][i]))]
            result[name] = dict(
                name=name,
                label=label(name),
                parts=parts,
                mass=float(d["mass"][i]),
                com=d["com"][i],
                inertia=d["inertia"][i],
                points=d[f"p{i}"],
                bounds=d[f"b{i}"],
                poses=[pose],
            )
    return sorted(result.values(), key=lambda o: o["name"])


def label(name: str) -> str:
    """'006_mustard_bottle' -> 'Mustard bottle', '065-a_cups' -> 'Cups (a)'."""
    number, words = name.split("_", 1)
    text = words.replace("_", " ").capitalize()
    return f"{text} ({number.split('-')[1]})" if "-" in number else text


def load_policy(path):
    """The actor network and the demonstration it edits."""
    with np.load(path) as d:
        weights = {k: torch.from_numpy(d[k]) for k in d.files if not k.startswith("demo/")}
        demo = {
            k[5:]: (d[k].item() if d[k].ndim == 0 else d[k])
            for k in d.files
            if k.startswith("demo/")
        }
    actor = Actor()
    actor.load_state_dict(weights)
    return actor.eval(), demo


# ---------------------------------------------------------------------------- policy


class PointNet(nn.Module):
    """Per point (mean xyz, xyz - mean) -> 128 -> 256 (ReLU); max over the first 128
    channels and mean over the last 128; then 256 -> 128 (ReLU)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.ReLU(), nn.Conv1d(128, 256, 1), nn.ReLU()
        )
        self.mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())

    def forward(self, xyz):  # (B, N, 3)
        mean = xyz.mean(1, keepdim=True)
        x = self.conv(torch.cat((mean.expand_as(xyz), xyz - mean), -1).transpose(1, 2))
        return self.mlp(torch.cat((x[:, :128].amax(-1), x[:, 128:].mean(-1)), -1))


class Actor(nn.Module):
    """Observation -> action: tanh of the mean of the trained Gaussian policy (no sampling)."""

    def __init__(self):
        super().__init__()
        self.pointnet = PointNet()
        sizes = [STATE_DIM + 128, 1024, 1024, 512, 512]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ELU()]
        self.actor = nn.Sequential(*layers, nn.Linear(512, 12))

    @torch.no_grad()
    def forward(self, obs):
        cloud = obs[:, STATE_DIM:].reshape(len(obs), POINTS, 3)
        return torch.tanh(self.actor(torch.cat((obs[:, :STATE_DIM], self.pointnet(cloud)), -1)))


# ---------------------------------------------------------------------------- kinematics


class LeftArm:
    """Batched FK/IK of the left arm's palm point (float64), from LW-v0's scene MJCF.
    Positions are in the environment frame (the arm-base frame of LW-v0)."""

    def __init__(self, device):
        model = mujoco.MjModel.from_xml_path(str(assets.path("lw_v0.xml")))
        body = model.body(f"left/{workstation.PALM_BODY['left']}").id
        chain = []
        while body:
            chain.append(body)
            body = model.body_parentid[body]
        f64 = dict(dtype=torch.float64, device=device)
        self.links = []  # (offset, rotation, joint axis or None) per body, base first
        for b in reversed(chain):
            joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
            axis = None
            if joints:
                (j,) = joints
                assert (
                    model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not model.jnt_pos[j].any()
                )
                axis = torch.tensor(model.jnt_axis[j], **f64)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[b])
            rotation = torch.tensor(rotation.reshape(3, 3), **f64)
            self.links.append((torch.tensor(model.body_pos[b], **f64), rotation, axis))
        ids = [model.joint(f"left/{name}").id for name in workstation.ARM_JOINTS["left"]]
        self.lower = torch.tensor(model.jnt_range[ids, 0], **f64)
        self.upper = torch.tensor(model.jnt_range[ids, 1], **f64)
        self.offset = torch.tensor(workstation.PALM_OFFSET, **f64)

    def forward(self, q, jacobian=False):
        """Palm point position (..., 3) and rotation (..., 3, 3) for joints q (..., 7); with
        ``jacobian`` also the geometric Jacobian (..., 6, 7)."""
        q = q.to(torch.float64)
        shape = q.shape[:-1]
        p = torch.zeros(*shape, 3, dtype=q.dtype, device=q.device)
        r = torch.eye(3, dtype=q.dtype, device=q.device).expand(*shape, 3, 3)
        origins, axes, k = [], [], 0
        for offset, rotation, axis in self.links:
            p = p + (r @ offset[:, None])[..., 0]
            r = r @ rotation
            if axis is not None:
                r = r @ _rotation(axis, q[..., k])
                origins.append(p)
                axes.append((r @ axis[:, None])[..., 0])
                k += 1
        p = p + (r @ self.offset[:, None])[..., 0]
        if not jacobian:
            return p, r
        z = torch.stack(axes, -1)
        lever = p[..., None] - torch.stack(origins, -1)
        return p, r, torch.cat((torch.linalg.cross(z, lever, dim=-2), z), -2)

    def inverse(self, p_goal, r_goal, seed, joints=7, damping=1e-4, step=0.2):
        """Damped least squares from ``seed`` (..., 7), each iteration projected into the real
        limits (box and wrist octagon). ``r_goal=None``: position only; ``joints=5``: J6, J7
        keep the seed's values. Returns joints, position and orientation errors."""
        q = seed.to(torch.float64).clone()
        shape = q.shape[:-1]
        rows = 6 if r_goal is not None else 3
        eye = torch.eye(rows, dtype=q.dtype, device=q.device)
        for _ in range(IK_ITERATIONS):
            p, r, jac = self.forward(q, jacobian=True)
            error = p_goal - p
            if r_goal is not None:
                error = torch.cat((error, _rotation_error(r_goal, r)), -1)
            jac = jac[..., :rows, :joints]
            jjt = jac @ jac.transpose(-1, -2) + damping * eye
            dq = (jac.transpose(-1, -2) @ torch.linalg.solve(jjt, error[..., None]))[..., 0]
            q = torch.cat((q[..., :joints] + dq.clamp(-step, step), q[..., joints:]), -1)
            q = project(q.reshape(-1, 7).contiguous(), self.lower, self.upper)[0].reshape(*shape, 7)
        p, r = self.forward(q)
        angle = (
            _rotation_error(r_goal, r).norm(dim=-1)
            if r_goal is not None
            else torch.zeros_like(p[..., 0])
        )
        return q, (p_goal - p).norm(dim=-1), angle


def _rotation(axis, angle):
    """Rotation matrices (..., 3, 3) about unit axes (..., 3) by angles (...) (Rodrigues)."""
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    k = torch.stack(
        (
            torch.stack((zero, -z, y), -1),
            torch.stack((z, zero, -x), -1),
            torch.stack((-y, x, zero), -1),
        ),
        -2,
    )
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return torch.eye(3, dtype=axis.dtype, device=axis.device) + s * k + (1 - c) * (k @ k)


def _rotation_error(r_goal, r):
    """Rotation vector (..., 3) taking ``r`` to ``r_goal``, world frame."""
    return axis_angle_from_quat(quat_from_matrix(r_goal @ r.transpose(-1, -2)))


def _smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * x * (10 - 15 * x + 6 * x * x)


# ---------------------------------------------------------------------------- quaternions
# Isaac Lab 3.0.0's functions (isaaclab/utils/math.py; BSD-3-Clause, see NOTICE), copied
# without TorchScript: the policy was trained on their results, sign conventions included
# (x, y, z, w; quat_from_matrix picks the best-conditioned candidate, not w >= 0).


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1).view(shape)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}.")
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(shape)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz, vec = torch.broadcast_tensors(quat[..., :3], vec)
    t = xyz.cross(vec, dim=-1) * 2
    return vec + quat[..., 3:4] * t + xyz.cross(t, dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2], dim=-1),
            torch.stack([q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20], dim=-1),
            torch.stack([m20 + m02, m21 + m12, q_abs[..., 3] ** 2, m10 - m01], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    quat = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    invalid = (quat.norm(p=2, dim=-1, keepdim=True) - 1.0).abs() > 2e-5
    return torch.where(invalid, torch.full_like(quat, float("nan")), quat)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    mag = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., :3] / sin_half_angles_over_angles.unsqueeze(-1)


# ---------------------------------------------------------------------------- station


def parking(i: int) -> tuple[float, float, float]:
    x, y, z = PARKING["origin"]
    row, column = divmod(i, PARKING["columns"])
    return (x + PARKING["spacing"] * column, y + PARKING["spacing"] * row, z)


def merged(parts) -> tuple[np.ndarray, np.ndarray]:
    """The parts as one mesh: a viewer draws each mesh separately, and dozens of objects of up to
    64 pieces each would slow it to a fifth of real time."""
    offsets = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    return np.concatenate([v for v, _ in parts]), np.concatenate(
        [f + o for (_, f), o in zip(parts, offsets)]
    )


def station(items: list[dict]) -> workstation.Station:
    """One LW-v0 station with the left arm streaming points (``interpolation=False``), the
    floor, and every object parked (each a free object ``item_NN`` whose initial pose is its
    parking spot, so the station's reset puts it back there)."""
    return workstation.Station(
        objects={
            f"item_{i:02d}": objects.object_cfg(
                o["parts"], o["mass"], com=o["com"], inertia=o["inertia"], pos=parking(i), visual=merged(o["parts"])
            )
            for i, o in enumerate(items)
        },
        arm_left=workstation.arm_action_cfg("left", interpolation=False),
    )


# ---------------------------------------------------------------------------- scene


class GraspScene:
    """The interactive episode on a :func:`station`. :meth:`press` takes the panel's buttons
    (``"drop"`` with an object index, ``"grasp"``, ``"reset"``); :meth:`step` advances one
    decision.

    States: ``empty`` -> Drop -> ``settling`` -> ``ready`` -> Grasp -> ``grasping`` ->
    ``done`` (result in :attr:`message`); Reset (or Drop) from any state."""

    def __init__(self, station: workstation.Station, items: list[dict], actor, demo: dict):
        self.station, self.items, self.actor, self.demo = station, items, actor, demo
        self.device = torch.device("cpu")
        model = station.model
        self.objects = [station.objects[f"item_{i:02d}"] for i in range(len(items))]
        self.hand_action = station.actions["hand_left"]
        joints = [model.joint(f"left/{n}").id for n in (*workstation.ARM_JOINTS["left"], *workstation.HAND_JOINTS["left"])]
        self._qpos, self._qvel = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        # The left hand: its base link and fingers (linker-lab's contact sensor filter).
        self._hand_bodies = frozenset(b for b in range(model.nbody) if model.body(b).name.startswith("left/hand_lh_"))
        self._bodies = [model.body(f"item_{i:02d}").id for i in range(len(items))]
        for body, o in zip(self._bodies, items):
            if abs(float(model.body_mass[body]) - o["mass"]) > 1e-4 * o["mass"]:
                raise RuntimeError("the station does not hold the expected objects")
        self.table_top = objects.table()["top"]
        self.rng = np.random.default_rng()

        f64 = dict(device=self.device, dtype=torch.float64)
        self.demo_q = torch.tensor(demo["arm"], **f64)
        self.demo_hand = torch.tensor(demo["hand"], **f64)
        self.anchor = torch.tensor(demo["object_pos"], **f64)
        self.region = tuple(float(v) for v in demo["region"])
        self.demo_object_rot = matrix_from_quat(torch.tensor(demo["object_quat"], **f64))
        self.kin = LeftArm(self.device)
        self.demo_palm_p, self.demo_palm_r = self.kin.forward(self.demo_q)
        limits = torch.tensor(model.jnt_range[joints], **f64)
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        names = (*workstation.ARM_JOINTS["left"], *workstation.HAND_JOINTS["left"])
        self.home = torch.tensor([workstation.HOME["left"][n] for n in names], **f64)[None]
        self.scale = torch.tensor(ACTION_SCALE, **f64)
        self._next_park = 0
        self.choice = 0  # index of the object chosen in the panel's list
        self.active = None  # index of the object on the table
        self.state, self.message, self.pending = "empty", "Pick an object and press Drop.", None

    # -- buttons

    def press(self, button: str, item: int | None = None):
        """Queue a button press; it takes effect at the next :meth:`step`."""
        self.pending = (button, item)

    def _reset(self):
        self.station.reset()  # robot home, every object back to its parking spot
        self.active = None
        self.state, self.message = "empty", "Pick an object and press Drop."

    def _drop(self, item: int):
        self._reset()
        self.active, self.attempt = item, 0
        o = self.items[item]
        f64 = dict(device=self.device, dtype=torch.float64)
        self.points = torch.tensor(o["points"], device=self.device)[None]
        self.bounds_mid = torch.tensor(o["bounds"], **f64).mean(0)[None]
        k = int(self.rng.integers(len(o["poses"])))
        self.rest_quat = torch.tensor(o["poses"][k][0], **f64)[None]
        self.rest_height = o["poses"][k][1]
        self.pose_text = f"resting pose {k + 1} of {len(o['poses'])}"
        self._place()

    def _place(self):
        f64 = dict(device=self.device, dtype=torch.float64)
        yaw = torch.tensor([self.rng.uniform(-math.pi, math.pi)], **f64)
        x0, x1, y0, y1 = self.region
        centre = torch.tensor([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)], **f64)
        zero = torch.zeros_like(yaw)
        quat = quat_mul(quat_from_euler_xyz(zero, zero, yaw), self.rest_quat)
        pos = torch.zeros(1, 3, **f64)
        pos[0, :2] = centre - quat_apply(quat, self.bounds_mid)[0, :2]
        pos[0, 2] = self.table_top + self.rest_height + DROP_CLEARANCE
        self.placed_xy = pos[:, :2]
        self.objects[self.active].set_pose(pos[0].numpy(), quat[0].numpy())
        self.tick, self.quiet, self.window = 0, 0, []
        self.state = "settling"
        self.message = f"{self.items[self.active]['label']} ({self.pose_text}): settling..."

    def _grasp(self):
        self.settled_pos, self.settled_quat = self._object_pose()
        self.arm, self.hand, self.plan_valid = self.plan(self.actor(self.observe()))
        if not self.plan_valid[0]:  # not executed: the arm stays home
            self.arm[:], self.hand[:] = self.home[:, None, :7], self.home[:, None, 7:]
        self.k = self.demo["settle_end"]
        self.ok = {"finite": True, "in range": True, "risen": True, "alone": True}
        self.state, self.message = "grasping", "Grasping: approach"

    # -- one decision

    def step(self):
        if self.pending is not None:
            (button, item), self.pending = self.pending, None
            if button == "reset":
                self._reset()
            elif button == "drop":
                self._drop(item)
            elif button == "grasp" and self.state == "ready":
                self._grasp()
        if self.state == "grasping":
            arm, hand = self._grasp_targets()
        elif self.state == "done":
            arm, hand = self.lift[:, -1], self.hand_hold
        else:
            arm, hand = self.home[:, :7], self.home[:, 7:]
        # Park one waiting object per decision, in turn: they all fall together, 0.6 m apart
        # with nothing below, so each only needs putting back every len(items) decisions.
        self._next_park = (self._next_park + 1) % len(self.objects)
        if self._next_park != self.active:
            self.objects[self._next_park].set_pose(parking(self._next_park))
        self.station.step(arm_left=arm[0].numpy(), hand_left=hand[0].numpy())
        if self.state == "settling":
            self._settle_check()
        elif self.state == "grasping":
            self._grasp_check()

    def _settle_check(self):
        self.tick += 1
        self.window = (self.window + [self._object_pose()])[-(WINDOW + 1) :]
        if len(self.window) > WINDOW:
            (p0, q0), (p1, q1) = self.window[0], self.window[-1]
            linear = (p1 - p0).norm(dim=-1) / (WINDOW / 60)
            angular = axis_angle_from_quat(quat_mul(q1, quat_inv(q0))).norm(dim=-1) / (WINDOW / 60)
            self.quiet = self.quiet + 1 if bool((linear < 0.01) & (angular < 0.1)) else 0
        if self.tick < SETTLE_TICKS:
            return
        pos, quat = self._object_pose()
        finite = bool(torch.isfinite(pos).all() & torch.isfinite(quat).all())
        near = float((pos[:, :2] - self.placed_xy).norm()) < 0.08
        if finite and float(pos[0, 2]) > self.table_top and near and self.quiet >= QUIET_TICKS:
            self.state = "ready"
            self.message = (
                f"{self.items[self.active]['label']} ({self.pose_text}) is at rest. Press Grasp."
            )
        elif self.attempt < RETRIES:
            self.attempt += 1
            self._place()
        else:
            self.state, self.message = "empty", "The object did not come to rest. Press Drop again."

    def _grasp_targets(self):
        d = self.demo
        self.k += 1
        if self.k == d["lift_start"]:
            q_now = self._joints()[:, :7]
            self.lift, lift_valid = self.plan_lift(q_now)
            if not (lift_valid[0] and self.plan_valid[0]):
                self.lift[:] = q_now[:, None]
            self.plan_valid &= lift_valid
            self.hand_hold = self.hand[:, d["close_end"]].clone()
        if self.k < d["lift_start"]:
            phase = "approach" if self.k < d["close_start"] else "closing the hand"
            self.message = f"Grasping: {phase}"
            return self.arm[:, self.k], self.hand[:, self.k]
        self.message = "Grasping: lift" if self.k < d["hold_start"] else "Grasping: hold"
        return self.lift[:, min(self.k - d["lift_start"], self.lift.shape[1] - 1)], self.hand_hold

    def _grasp_check(self):
        """After each grasping decision: the checks, and the result at the end."""
        d = self.demo
        q = self._joints()
        pos, quat = self._object_pose()
        self.ok["finite"] &= bool(
            torch.isfinite(q).all() & torch.isfinite(pos).all() & torch.isfinite(quat).all()
        )
        tolerance = JOINT_LIMIT_TOLERANCE
        self.ok["in range"] &= bool(
            ((q >= self.lower - tolerance) & (q <= self.upper + tolerance)).all()
        )
        if self.k >= d["hold_start"]:
            self.ok["risen"] &= float(pos[0, 2] - self.settled_pos[0, 2]) >= RISE
            self.ok["alone"] &= float(np.linalg.norm(self.other_contact_force())) < CONTACT_TOLERANCE
        if self.k < d["end"]:
            return
        failures = {
            "finite": "the simulation became unstable",
            "in range": "a joint left its range",
            "risen": "the object was not lifted 10 cm",
            "alone": "the object touched something besides the hand",
        }
        if not self.plan_valid[0]:
            reason = "no feasible grasp plan within the arm's limits"
        else:
            reason = next((text for key, text in failures.items() if not self.ok[key]), None)
        name = self.items[self.active]["label"]
        self.message = f"{name}: grasped." if reason is None else f"{name}: missed, {reason}."
        self.state = "done"

    # -- state

    def _object_pose(self):
        pos, quat = self.objects[self.active].pose()
        f64 = dict(device=self.device, dtype=torch.float64)
        return torch.tensor(pos, **f64)[None], torch.tensor(quat, **f64)[None]

    def _joints(self):
        return torch.tensor(self.station.data.qpos[self._qpos], dtype=torch.float64)[None]

    def other_contact_force(self) -> np.ndarray:
        """The net force (3,) [N], station frame, of the last physics step's contacts on the
        object on the table from anything but the left hand (linker-lab: the contact
        sensor's net force minus its force from the hand)."""
        model, data = self.station.model, self.station.data
        body, total, force = self._bodies[self.active], np.zeros(3), np.zeros(6)
        for i, contact in enumerate(data.contact[: data.ncon]):
            b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            if body not in (b1, b2) or contact.efc_address < 0:
                continue
            if (b2 if b1 == body else b1) in self._hand_bodies:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            # The force in the contact frame (rows: normal from geom1 to geom2, then two
            # tangents) is the one geom1 exerts on geom2.
            world = contact.frame.reshape(3, 3).T @ force[:3]
            total += world if b2 == body else -world
        return total

    # -- the policy's observation and plan (as trained)

    def observe(self):
        """The policy's observation (1, 1,595) of the scene now: :meth:`observation` of the
        left unit's joints, its hand's motor shafts and the object's velocity."""
        linear, angular = self.objects[self.active].velocity()
        qd = torch.tensor(self.station.data.qvel[self._qvel], dtype=torch.float64)[None]
        vel = torch.tensor(np.concatenate((linear, angular)), dtype=torch.float64)[None]
        return self.observation(self._joints(), qd, self.hand_action.shaft.double(), vel)

    def observation(self, q, qd, shaft, vel):
        """The observation (1, 1,595) for the left arm and hand joints' positions ``q`` and
        velocities ``qd`` (1, 13), the hand's motor shafts (1, 6) and the object's velocity
        ``vel`` (1, 6: its centre of mass, then angular), with the object at its settled pose
        (``settled_pos``, ``settled_quat``)."""
        span = self.upper - self.lower
        q_norm = 2 * (q - self.lower) / span - 1
        shaft = 2 * (shaft - self.lower[7:]) / span[7:] - 1
        pos, quat = self.settled_pos, self.settled_quat
        palm_p, palm_r = self.kin.forward(q[:, :7])
        inv = quat_inv(quat)
        palm_rel = quat_apply(inv, palm_p - pos) / 0.5
        palm_quat = quat_mul(inv, quat_from_matrix(palm_r))
        k = self.demo["close_start"]
        center = pos + quat_apply(quat, self.bounds_mid)
        grasp_rel = quat_apply(inv, self.demo_palm_p[k] + center - self.anchor - pos) / 0.5
        grasp_quat = quat_mul(inv, quat_from_matrix(self.demo_palm_r[k])[None])
        cloud = (matrix_from_quat(quat).float() @ self.points.transpose(1, 2)).transpose(
            1, 2
        ) / 0.25
        state = torch.cat(
            (
                q_norm,
                qd,
                shaft,
                (pos - self.anchor) / 0.1,
                quat,
                vel,
                palm_rel,
                palm_quat,
                grasp_rel,
                grasp_quat,
            ),
            -1,
        )
        assert state.shape[1] == STATE_DIM
        return torch.cat((state.float(), cloud.reshape(1, -1)), -1)

    def plan(self, actions):
        """Arm joint targets (1, T, 7), shaft targets (1, T, 6) for ticks 0 .. lift_start - 1,
        and whether the plan is valid (1,)."""
        d = self.demo
        a = actions.double().clamp(-1, 1) * self.scale
        rd = self.demo_object_rot
        r_edit = matrix_from_quat(quat_from_euler_xyz(a[:, 3], a[:, 4], a[:, 5]))  # Rz Ry Rx
        w = axis_angle_from_quat(quat_from_matrix(rd @ r_edit @ rd.T))  # about the demo object
        center = self.settled_pos + quat_apply(self.settled_quat, self.bounds_mid)
        translation = center - self.anchor + (rd @ a[:, :3, None])[..., 0]

        ticks = torch.arange(d["settle_end"], d["lift_start"], device=self.device)
        blend = _smoothstep((ticks - d["settle_end"]) / (d["blend_end"] - d["settle_end"]))
        angle = w.norm(dim=-1)[:, None] * blend[None]  # (1, T)
        axis = (w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12))[:, None]
        rb = _rotation(axis, angle)  # (1, T, 3, 3)
        p_demo, r_demo = self.demo_palm_p[ticks], self.demo_palm_r[ticks]
        p_goal = self.anchor + (rb @ (p_demo - self.anchor)[None, ..., None])[..., 0]
        p_goal = p_goal + blend[None, :, None] * translation[:, None]
        seed = self.demo_q[ticks][None]
        q, ep, er = self.kin.inverse(p_goal, rb @ r_demo, seed)
        step = (q[:, 1:] - q[:, :-1]).abs().amax((1, 2))
        valid = (ep.amax(1) <= PLAN_POSITION_TOLERANCE) & (er.amax(1) <= PLAN_ROTATION_TOLERANCE)
        valid &= (step <= PLAN_MAX_STEP) & (
            (q[:, 0] - self.demo_q[ticks[0]]).abs().amax(-1) <= PLAN_MAX_STEP
        )

        arm = self.demo_q[: d["lift_start"]][None].clone()
        arm[:, ticks] = q
        all_ticks = torch.arange(d["lift_start"], device=self.device)
        close = _smoothstep((all_ticks - d["close_start"]) / (d["close_end"] - d["close_start"]))
        hand = (
            self.demo_hand[all_ticks.clamp(max=d["close_end"])][None]
            + close[None, :, None] * a[:, None, 6:]
        )
        hand = torch.minimum(torch.maximum(hand, self.lower[7:]), self.upper[7:])
        return arm, hand, valid

    def plan_lift(self, q_now):
        """Joint targets raising the palm point straight up from ``q_now`` with J1-J5 (the
        wrist held), and whether the IK reached every position."""
        steps = self.demo["hold_start"] - self.demo["lift_start"]
        p0, _ = self.kin.forward(q_now)
        s = _smoothstep(torch.arange(steps + 1, device=self.device, dtype=torch.float64) / steps)
        p_goal = p0[:, None].repeat(1, steps + 1, 1)
        p_goal[..., 2] += LIFT_HEIGHT * s
        q, ep, _ = self.kin.inverse(
            p_goal, None, q_now[:, None].expand(-1, steps + 1, -1), joints=5
        )
        valid = (ep.amax(1) <= PLAN_POSITION_TOLERANCE) & (
            (q[:, 1:] - q[:, :-1]).abs().amax((1, 2)) <= PLAN_MAX_STEP
        )
        return q, valid


# ---------------------------------------------------------------------------- panel


def make_panel(scene: GraspScene) -> Panel:
    """The panel: the object list (its choice in ``scene.choice``), the three buttons, the
    scene's message."""
    panel = Panel("ycb-grasp")
    panel.select("Object", [o["label"] for o in scene.items], lambda i: setattr(scene, "choice", i))
    panel.button("Drop", lambda: scene.press("drop", scene.choice))
    panel.button("Grasp", lambda: scene.press("grasp"))
    panel.button("Reset", lambda: scene.press("reset"))
    panel.text(lambda: scene.message)
    return panel


# ---------------------------------------------------------------------------- entry point


def scene(artifacts: dict) -> GraspScene:
    """The scene on its station, with the policy, from the entry's artifacts (file name ->
    path)."""
    items = load_objects(artifacts["objects.npz"])
    actor, demo = load_policy(artifacts["policy.npz"])
    return GraspScene(station(items), items, actor, demo)


def run(artifacts: dict, viewer: str = "studio", port: int = 0) -> viewers.Stats:
    """Open the scene in ``viewer`` (``"studio"`` or ``"viser"``); returns when stopped
    (Ctrl+C)."""
    s = scene(artifacts)
    camera = (CAMERA["eye"], CAMERA["lookat"])
    return viewers.run(s.station, make_panel(s), step=s.step, viewer=viewer, port=port, camera=camera)
