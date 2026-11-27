"""The playground: drive LW-v0 by hand in a viewer (``linker-sim play``).

The scene is a :class:`~linker_sim.workstation.Station` with a 5 cm box on the table and
the floor. You move a goal for the palm point (``workstation.PALM_OFFSET``) of one arm at
a time; the arm follows through ``ArmAction(command="pose", frame="palm")`` with
interpolation on: it moves as the real arm does in the xCore SDK's tracking mode, within
the real controller's limits.

=====================  ==============================================================
I/K, J/L, U/O          move the goal forward/back, left/right, up/down (relative to
                       the camera's view)
Shift + the same keys  turn the goal about the palm point (I/K pitch, J/L yaw, U/O roll)
Tab                    switch arm
1 / 2 / 3 / 4          hand open / close / pinch / thumb oppose (``hand.PRESETS``)
R                      reset
Mouse                  turn and move the camera; in viser also drag the goal's handle,
                       and Ctrl + right drag pulls a body
=====================  ==============================================================

Keys move the goal like a jog on a teach pendant: at :data:`JOG_SPEED` while held, and
never more than :data:`LEAD` ahead of the palm point, so the arm stops soon after the key
is released (the arm's own speed is set by its joint limits and is often below the jog
speed). viser reports key presses, not held keys: there each press moves the goal by one
tap (0.1 s of jog: 1 cm or 3 degrees), and holding a key repeats it at the keyboard's
repeat rate. The panel shows the arm TCP pose (the frame the real controller reports and
most learned policies command), how far the palm point is from its goal, limit
interventions, contacts and the simulation speed. The goal is drawn as axes, each arm's
TCP as axes and its palm point as a yellow sphere.

Copied from linker-lab v1.0.0:src/linker_lab/playground.py: :data:`JOG_SPEED`,
:data:`LEAD`, :data:`BOX`, :data:`CAMERA`, :data:`KEYS`, :data:`ACTIONS`,
:func:`camera_axes` and :func:`jog`; change them there first (``tools/check_copies.py``
compares them). New here: the playground is a ``linker_sim.viewer.Panel`` around a
station, so it runs in either viewer.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from linker_sim import hand, objects, physics, viewer, workstation
from linker_sim.viewer import Axes, Panel, Sphere

JOG_SPEED = (0.1, 0.5)
"""Speed of the goal while a key is held: [m/s], [rad/s]."""
LEAD = (0.05, 0.3)
"""Keys move the goal at most this far from the palm point: [m], [rad]."""
BOX = {"edge": 0.05, "mass": 0.1, "xy": (0.42, -0.25)}
"""The box on the table: edge [m], mass [kg], position [m], below the right hand."""
CAMERA = {"eye": (1.2, -0.85, 0.2), "lookat": (0.3, 0.0, -0.35)}
"""Initial viewer camera [m], environment frame: in front of the table, to the right."""

KEYS = ("i", "k", "j", "l", "u", "o")
ACTIONS = {
    "tab": "switch",
    "1": "open",
    "2": "close",
    "3": "pinch",
    "4": "thumb_oppose",
    "r": "reset",
}

LABELS = {
    "switch": "Switch arm",
    "open": "Open",
    "close": "Close",
    "pinch": "Pinch",
    "thumb_oppose": "Thumb oppose",
    "reset": "Reset",
}
"""The panel's button for each of :data:`ACTIONS`."""


def camera_axes(front) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward (the view direction, horizontal), right and up for keys relative to a camera
    looking along ``front``."""
    up = np.array([0.0, 0.0, 1.0])
    forward = np.array([front[0], front[1], 0.0])
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])  # looking straight down
    return forward, np.cross(forward, up), up


def jog(goal, palm, keys: set[str], front, dt: float):
    """The goal after ``dt`` with ``keys`` held (``"shift"`` turns instead of moving).

    ``goal`` and ``palm`` are (position (3,), rotation matrix (3, 3)) in the world frame. A
    step that would put the goal more than :data:`LEAD` from the palm point is not taken,
    unless it brings the goal closer.
    """
    forward, right, up = camera_axes(front)
    p, R = goal
    if "shift" in keys:
        axes = {"i": right, "k": -right, "j": up, "l": -up, "u": forward, "o": -forward}
        w = sum((axes[k] for k in KEYS if k in keys), np.zeros(3))
        if not w.any():
            return goal
        R_new = Rotation.from_rotvec(w * JOG_SPEED[1] * dt).as_matrix() @ R
        angle = lambda Ra: Rotation.from_matrix(Ra @ palm[1].T).magnitude()  # noqa: E731
        if angle(R_new) > LEAD[1] and angle(R_new) >= angle(R):
            return goal
        return p, R_new
    axes = {"i": forward, "k": -forward, "l": right, "j": -right, "u": up, "o": -up}
    v = sum((axes[k] for k in KEYS if k in keys), np.zeros(3))
    if not v.any():
        return goal
    p_new = p + v * JOG_SPEED[0] * dt
    distance = lambda pa: np.linalg.norm(pa - palm[0])  # noqa: E731
    if distance(p_new) > LEAD[0] and distance(p_new) >= distance(p):
        return goal
    return p_new, R


def station() -> workstation.Station:
    """The playground's station: palm-point pose commands for both arms, the box on the
    table and the floor."""
    box = trimesh.creation.box((BOX["edge"],) * 3)
    parts = [(box.vertices, box.faces)]
    return workstation.Station(
        objects={"box": objects.object_cfg(parts, BOX["mass"], pos=objects.on_table(parts, BOX["xy"]))},
        arm_left=workstation.arm_action_cfg("left", command="pose", frame="palm"),
        arm_right=workstation.arm_action_cfg("right", command="pose", frame="palm"),
    )


class Playground:
    """The playground's state and panel around a station made by :func:`station`.

    :attr:`panel` is the interface; call :meth:`step` once per decision (``viewer.run``
    does: ``viewer.run(playground.station, playground.panel, step=playground.step)``).
    """

    def __init__(self, station: workstation.Station):
        self.station = station
        self.active = "right"
        """The arm the keys move."""
        self.goal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        """Palm-point goal per arm: position (3,), rotation matrix (3, 3), station frame."""
        self.hand: dict[str, np.ndarray] = {}
        """Shaft targets (6,) per hand."""
        self.peak_contacts = 0
        model = station.model
        self._tcp = {s: model.body(f"{s}/{workstation.TCP_BODY[s]}").id for s in workstation.SIDES}
        self.reset()
        self.panel = self._make_panel()

    # -- state

    def palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the palm point: position (3,), rotation matrix (3, 3), station frame."""
        pos, quat = self.station.palm_pose(side)
        return pos, Rotation.from_quat(quat).as_matrix()

    def reset(self) -> None:
        """The station at home, the goals at the palm points, the hands' targets at home."""
        self.station.reset()
        for side in workstation.SIDES:
            self.goal[side] = self.palm(side)
            self.hand[side] = np.array([workstation.HOME[side][n] for n in workstation.HAND_JOINTS[side]])
        self.peak_contacts = 0

    def reach_error(self, side: str) -> tuple[float, float]:
        """Distance [m] and angle [rad] from the palm point to its goal."""
        (p, R), (gp, gR) = self.palm(side), self.goal[side]
        return float(np.linalg.norm(gp - p)), float(Rotation.from_matrix(gR @ R.T).magnitude())

    def tcp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Arm TCP in the arm's base frame: position [m], extrinsic XYZ Euler angles [rad]
        (the xCore SDK's convention)."""
        data, body = self.station.data, self._tcp[side]
        w, x, y, z = data.xquat[body]
        pos, quat = self.station.to_base(side, data.xpos[body], (x, y, z, w))
        return pos, Rotation.from_quat(quat).as_euler("xyz")

    # -- input

    def act(self, name: str) -> None:
        """Do one of :data:`ACTIONS` (by name) for the active arm."""
        side = self.active
        if name == "switch":
            self.active = "left" if side == "right" else "right"
        elif name == "reset":
            self.reset()
        elif name == "thumb_oppose":
            self.hand[side] = hand.thumb_oppose(torch.as_tensor(self.hand[side])[None])[0].numpy()
        else:
            self.hand[side] = np.array(hand.PRESETS[name])

    def move_goal(self, pos, quat) -> None:
        """Put the active arm's goal at a pose (the drag handle)."""
        self.goal[self.active] = (np.asarray(pos, dtype=float), Rotation.from_quat(quat).as_matrix())

    def step(self) -> None:
        """One decision: jog the goal with the keys held, then step the station."""
        side, keys = self.active, set(self.panel.held)
        if keys & set(KEYS):
            self.goal[side] = jog(self.goal[side], self.palm(side), keys, self.panel.view, 1 / physics.DECISION_HZ)
        commands = {}
        for s in workstation.SIDES:
            p, R = self.goal[s]
            commands[f"arm_{s}"] = np.concatenate(self.station.to_base(s, p, Rotation.from_matrix(R).as_quat()))
            commands[f"hand_{s}"] = self.hand[s]
        self.station.step(**commands)
        self.peak_contacts = max(self.peak_contacts, int(self.station.data.ncon))

    # -- panel

    def _make_panel(self) -> Panel:
        panel = Panel("LW-v0 playground")
        panel.text(lambda: f"Arm: {self.active}   (Tab switches)")
        panel.text(lambda: "Arm TCP, base frame (as the SDK reports it)")

        def tcp_line(i: int) -> str:
            pos, euler = self.tcp_pose(self.active)
            if i == 0:
                return f"  x y z  {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m"
            return f"  rx ry rz  {euler[0]:+.3f} {euler[1]:+.3f} {euler[2]:+.3f} rad"

        panel.text(lambda: tcp_line(0))
        panel.text(lambda: tcp_line(1))

        def reach() -> str:
            distance, angle = self.reach_error(self.active)
            return f"Palm point to goal: {distance * 1000:.0f} mm, {np.degrees(angle):.0f} deg"

        panel.text(reach)
        hits = lambda: int(self.station.actions[f"arm_{self.active}"].limit_hits[0])  # noqa: E731
        panel.text(lambda: f"Limit interventions: {hits()} (since reset)")
        panel.text(lambda: f"Contacts: {self.station.data.ncon} (peak {self.peak_contacts})")
        for key, name in ACTIONS.items():
            panel.button(LABELS[name], lambda name=name: self.act(name), key=key)
        panel.text(lambda: "I/K J/L U/O  move the goal (camera-relative)")
        panel.text(lambda: "Shift + keys  turn it about the palm point")
        panel.hold(*KEYS, "shift")
        panel.markers(self._markers)
        panel.handle(
            lambda: (self.goal[self.active][0], Rotation.from_matrix(self.goal[self.active][1]).as_quat()),
            self.move_goal,
        )
        return panel

    def _markers(self) -> list:
        p, R = self.goal[self.active]
        markers = [Axes(p, Rotation.from_matrix(R).as_quat(), 0.08)]
        data = self.station.data
        for side in workstation.SIDES:
            w, x, y, z = data.xquat[self._tcp[side]]
            markers.append(Axes(data.xpos[self._tcp[side]].copy(), (x, y, z, w), 0.1))
            markers.append(Sphere(self.station.palm_pose(side)[0], 0.008, (1.0, 0.85, 0.2, 1.0)))
        return markers


def run(viewer_name: str = "studio", port: int = 0) -> viewer.Stats:
    """Open the playground in a viewer; returns when stopped (Ctrl+C)."""
    playground = Playground(station())
    camera = (CAMERA["eye"], CAMERA["lookat"])
    return viewer.run(
        playground.station, playground.panel, step=playground.step, viewer=viewer_name, port=port, camera=camera
    )
