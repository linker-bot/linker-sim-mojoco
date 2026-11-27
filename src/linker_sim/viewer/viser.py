"""The viser front end: a panel in viser's web viewer (the backup viewer).

viser (``viser``) serves a page that draws the scene with three.js in the browser; mjviser
(``mjviser.ViserMujocoScene``) turns the station's model into viser meshes once and moves
them after every decision. The panel becomes viser's own widgets. viser calls their
callbacks on its own threads; they are queued and run on the station's thread at the
start of the next decision.

What differs from Studio (``docs/viewers.md``):

- **Held keys.** viser reports key presses only (its "commands" with a hotkey), not
  releases. So a press of a key the panel holds (:meth:`Panel.hold`) counts as that key
  held for :data:`TAP` seconds, and the keyboard's auto-repeat, while a key is held down,
  extends it. With Shift held, the press counts as the key and Shift.
- **Drag handle.** A panel's :meth:`Panel.handle` is drawn as viser's transform controls.
- **Mouse.** Clicking a body selects it; Ctrl (Cmd on a Mac) and a right drag pulls it,
  a left drag turns it (mjviser's perturbation: a force on that body, as in MuJoCo's
  own viewer). W/A/S/D/Q/E and the arrow keys move viser's camera.
- **Localhost only.** The server is bound to ``127.0.0.1``; reach it from another
  computer through an HTTPS reverse proxy.

Importing this module imports viser and mjviser but starts nothing.
"""

from __future__ import annotations

import contextlib
import copy
import io
import queue
import threading
import time

import mjviser
import numpy as np
import viser

from linker_sim.viewer import panel as panels

TAP = 0.1
"""Seconds a key press counts as held [s] (viser reports no key releases)."""


def _wxyz(quat) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in quat)
    return (w, x, y, z)


def _look(marker: panels.Marker) -> tuple:
    """What a marker looks like, apart from where it is."""
    if isinstance(marker, panels.Sphere):
        return (panels.Sphere, marker.radius, tuple(marker.rgba))
    return (panels.Axes, marker.length)


def _escape(line: str) -> str:
    """Markdown that shows ``line`` as it is."""
    special = "\\`*_{}[]()#+-.!|<>~"
    return "".join("\\" + c if c in special else c for c in line) or "&nbsp;"


class Viser:
    """Shows a panel and the station in viser, at ``http://127.0.0.1:port/``.

    ``port`` 0 takes the first free port from 8080. ``camera``: the initial view, (eye,
    look-at point) [m] in the station frame.
    """

    name = "viser"

    def __init__(self, panel: panels.Panel, *, port: int = 0, camera=None):
        self.panel = panel
        self.port = port
        self.camera = camera
        self._server: viser.ViserServer | None = None
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._taps: dict[str, float] = {}  # key -> wall time its tap ends
        self._lock = threading.Lock()
        self._dragged = None  # the handle's latest dragged pose, not yet handed on
        self._dragging = False

    def open(self, station) -> str:
        """Start the server and send it the station's scene; returns the page's address."""
        self._station = station
        with contextlib.redirect_stdout(io.StringIO()):  # its banner; run() prints the address
            server = self._server = viser.ViserServer(host="127.0.0.1", port=self.port or 8080)
        self.port = server.get_port()
        if self.camera is not None:
            eye, target = self.camera
            server.initial_camera.position = tuple(float(v) for v in eye)
            server.initial_camera.look_at = tuple(float(v) for v in target)
            server.initial_camera.up = (0.0, 0.0, 1.0)
        # mjviser adjusts the visual options of the model it is given: give it a copy.
        self._scene = mjviser.ViserMujocoScene(server, copy.deepcopy(station.model), num_envs=1)
        self._perturb = self._scene.enable_perturbation()
        self._scene.update_from_mjdata(station.data)
        self._build(server)
        return f"http://127.0.0.1:{self.port}/"

    def _build(self, server: viser.ViserServer) -> None:
        put = self._inbox.put
        p = self.panel
        self._lines = []
        with server.gui.add_folder(p.title):
            for element in p._elements:
                if isinstance(element, panels._Text):
                    self._lines.append([server.gui.add_markdown("&nbsp;"), ""])
                elif isinstance(element, panels._Button):
                    label = element.label + (f"  ({element.key})" if element.key else "")
                    button = server.gui.add_button(label)
                    button.on_click(lambda _, f=element.on_click: put(f))
                elif isinstance(element, panels._Select):
                    options = element.options
                    dropdown = server.gui.add_dropdown(
                        element.label, options, initial_value=options[element.index]
                    )
                    dropdown.on_update(
                        lambda e, f=element.on_select, o=options: put(lambda: f(o.index(e.target.value)))
                    )
                elif isinstance(element, panels._Slider):
                    slider = server.gui.add_slider(
                        element.label, min=element.low, max=element.high,
                        step=(element.high - element.low) / 1000, initial_value=element.value,
                    )  # fmt: skip
                    slider.on_update(lambda e, f=element.on_change: put(lambda: f(float(e.target.value))))
            self._status = server.gui.add_markdown("&nbsp;")
        # Keys: viser's commands, one per key (and per key with Shift, for held keys).
        hotkey = {"tab": "tab", "space": "space"}
        for key, on_press in p._keys.items():
            command = server.gui.add_command(f"Key {key}", hotkey=hotkey.get(key, key.upper()), modifier=None)
            command.on_trigger(lambda _, f=on_press: put(f))
        shifts = [None, "shift"] if "shift" in p._held else [None]
        for key in sorted(p._held - {"shift"} - set(p._keys)):
            for modifier in shifts:
                name = f"Hold {key}" + (" with Shift" if modifier else "")
                command = server.gui.add_command(name, hotkey=hotkey.get(key, key.upper()), modifier=modifier)
                command.on_trigger(lambda _, k=key, m=modifier: self._tap(k, m))
        # Markers and the drag handle.
        self._markers: list = []
        if p._handle is not None:
            pos, quat = p._handle.pose()
            controls = server.scene.add_transform_controls(
                "/panel/handle", scale=0.15, position=tuple(pos), wxyz=_wxyz(quat)
            )

            @controls.on_update
            async def _(event) -> None:  # async: runs on viser's event loop, in order
                with self._lock:
                    self._dragging = event.phase != "end"
                    self._dragged = (np.array(event.target.position), np.array(event.target.wxyz))

            self._controls = controls

    def _tap(self, key: str, modifier: str | None) -> None:
        end = time.perf_counter() + TAP
        with self._lock:
            self._taps[key] = end
            if modifier:
                self._taps["shift"] = end

    def is_open(self) -> bool:
        return self._server is not None

    def poll(self) -> None:
        """Run the callbacks of the input that arrived since the last call."""
        p = self.panel
        while True:
            try:
                callback = self._inbox.get_nowait()
            except queue.Empty:
                break
            callback()
        now = time.perf_counter()
        with self._lock:
            p.held = frozenset(k for k, end in self._taps.items() if end > now)
            dragged, self._dragged = self._dragged, None
        if dragged is not None:
            w, x, y, z = dragged[1]
            p._handle.on_move(dragged[0], np.array([x, y, z, w]))
        clients = list(self._server.get_clients().values())
        if clients:
            camera = clients[0].camera
            view = np.asarray(camera.look_at, dtype=float) - np.asarray(camera.position, dtype=float)
            if np.linalg.norm(view) > 1e-9:
                p.view = view / np.linalg.norm(view)
        self._perturb.step(self._station.data, paused=False)

    def show(self, lines: list[str], markers: list[panels.Marker], status: str) -> None:
        """Move the scene to the station's state; update text lines, markers and status."""
        self._scene.update_from_mjdata(self._station.data)
        for entry, line in zip(self._lines, lines):
            if entry[1] != line:
                entry[0].content, entry[1] = _escape(line), line
        if status and self._status.content != _escape(status):
            self._status.content = _escape(status)
        self._show_markers(markers)
        p = self.panel
        if p._handle is not None:
            with self._lock:
                dragging = self._dragging
            if not dragging:
                pos, quat = p._handle.pose()
                self._controls.position = tuple(float(v) for v in pos)
                self._controls.wxyz = _wxyz(quat)

    def _show_markers(self, markers: list[panels.Marker]) -> None:
        scene = self._server.scene
        looks = [_look(m) for m in markers]
        if looks != [_look(m) for m, _ in self._markers]:  # different markers: make them anew
            for _, handle in self._markers:
                handle.remove()
            self._markers = []
            for i, marker in enumerate(markers):
                if isinstance(marker, panels.Sphere):
                    *rgb, a = marker.rgba
                    handle = scene.add_icosphere(
                        f"/panel/marker_{i}", radius=marker.radius,
                        color=tuple(round(255 * c) for c in rgb), opacity=float(a),
                    )  # fmt: skip
                else:
                    handle = scene.add_frame(
                        f"/panel/marker_{i}", axes_length=marker.length, axes_radius=0.03 * marker.length
                    )
                self._markers.append((marker, handle))
        for i, marker in enumerate(markers):
            handle = self._markers[i][1]
            self._markers[i] = (marker, handle)
            handle.position = tuple(float(v) for v in marker.pos)
            if isinstance(marker, panels.Axes):
                handle.wxyz = _wxyz(marker.quat)

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
