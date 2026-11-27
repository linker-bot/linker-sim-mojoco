"""The MuJoCo Studio front end: a panel in Studio's web viewer (the default viewer).

Studio's web viewer (``mujoco.studio``) runs on its own thread in this process and serves
a page that draws the model with MuJoCo's own renderer, in the browser (WebGL). The
station's state goes to it once per decision. The panel is drawn with Studio's ImGui;
clicks, choices and keys come back as Studio events, and run the panel's callbacks on
the station's thread (inside ``ViewerHandle.sync``).

What is left out or changed, and why:

- **Studio's own interface** (``mujoco.studio.app``) is not loaded: it binds nearly every
  letter key (I toggles inertia boxes), so the panel could not have them, and its
  simulation controls would bypass the station's actions.
- **Localhost only.** Studio 4.0.1 serves plain HTTP on every network interface with no
  authentication, and has no option to change it yet (a TODO in its code). While the
  viewer starts, its socket is bound to ``127.0.0.1`` instead, and Studio's start-up
  banner, which would name the computer as a remote address, is replaced by one line with
  the local address. Reach it from another computer through an HTTPS reverse proxy.

Importing this module imports Studio but starts nothing.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import threading
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco.studio import launch_web, messages, plugin, viewer_protocol
from mujoco.studio.dear_imgui import dear_imgui as imgui
from mujoco.studio.web import web_server

from linker_sim.viewer import panel as panels

# Messages between the station's thread (sim side) and Studio's thread (viewer side).


@dataclass(frozen=True)
class _Ui(messages.Snapshot):
    """Sim -> viewer after every decision: text lines, markers and status (latest wins)."""

    lines: tuple[str, ...]
    markers: tuple
    status: str


@dataclass(frozen=True)
class _Input(messages.Event):
    """Viewer -> sim: element ``index`` clicked (``value`` None) or set to ``value``."""

    index: int
    value: float | None = None


@dataclass(frozen=True)
class _Press(messages.Event):
    """Viewer -> sim: ``key`` pressed."""

    key: str


@dataclass(frozen=True)
class _Keyboard(messages.Snapshot):
    """Viewer -> sim every frame: the keys held and the camera's view (latest wins)."""

    held: frozenset
    view: tuple[float, float, float]


# Helpers.


def camera_view(azimuth: float, elevation: float) -> np.ndarray:
    """The view direction of MuJoCo's free camera (angles in degrees)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _imgui_key(key: str):
    names = {"tab": "Tab", "space": "Space", "shift": "LeftShift"}
    return getattr(imgui.Key, names.get(key) or (f"N{key}" if key.isdigit() else key.upper()))


def _marker_geoms(marker: panels.Marker) -> list[mujoco.MjvGeom]:
    """MuJoCo's decoration geoms for one marker."""
    pos = np.asarray(marker.pos, dtype=float)
    if isinstance(marker, panels.Sphere):
        geom = mujoco.MjvGeom()
        size, rgba = np.array([marker.radius, 0.0, 0.0]), np.asarray(marker.rgba, dtype=np.float32)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, np.eye(3).ravel(), rgba)
        return [geom]
    x, y, z, w = marker.quat
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.array([w, x, y, z], dtype=float))
    geoms = []
    for axis in range(3):
        geom = mujoco.MjvGeom()
        rgba = np.array([axis == 0, axis == 1, axis == 2, 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), pos, np.eye(3).ravel(), rgba)
        end = pos + marker.length * rotation.reshape(3, 3)[:, axis]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, 0.03 * marker.length, pos, end)
        geoms.append(geom)
    return geoms


@contextlib.contextmanager
def _localhost_only():
    """While Studio's viewer starts: bind its socket to 127.0.0.1 and swallow its banner."""
    bind = web_server.bind_public_socket
    web_server.bind_public_socket = lambda host, port=0: bind("127.0.0.1", port)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        web_server.bind_public_socket = bind


# The two plugins.


class _Sim(plugin.SimPlugin):
    """Runs the panel's callbacks; Studio calls it from ``ViewerHandle.sync`` (our thread)."""

    def __init__(self, panel: panels.Panel):
        self.panel = panel

    @messages.handler
    def on_input(self, event: _Input) -> bool:
        element = self.panel._elements[event.index]
        if isinstance(element, panels._Button):
            element.on_click()
        elif isinstance(element, panels._Select):
            element.on_select(int(event.value))
        elif isinstance(element, panels._Slider):
            element.on_change(float(event.value))
        return True

    @messages.handler
    def on_press(self, event: _Press) -> bool:
        self.panel._keys[event.key]()
        return True

    @messages.handler
    def on_keyboard(self, snapshot: _Keyboard) -> bool:
        self.panel.held = snapshot.held
        self.panel.view = np.array(snapshot.view)
        return True


class _View(plugin.ViewerPlugin):
    """Draws the panel and the markers and reads the keys; runs on Studio's thread."""

    def __init__(self, front: Studio):
        self.front = front
        self.panel = front.panel
        self.ui = _Ui((), (), "")
        self.values = [getattr(e, "index", getattr(e, "value", None)) for e in self.panel._elements]

    @messages.handler
    def on_init(self, event: plugin.ViewerInitEvent) -> None:
        if self.front.camera is not None:
            eye, target = (np.asarray(v, dtype=float) for v in self.front.camera)
            camera = event.viewer.camera
            forward = target - eye
            camera.lookat[:] = target
            camera.distance = float(np.linalg.norm(forward))
            forward /= camera.distance
            camera.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
            camera.elevation = math.degrees(math.asin(forward[2]))
        self.front.port = event.viewer.port
        self.front._ready.set()

    @messages.handler
    def on_ui(self, snapshot: _Ui) -> None:
        self.ui = snapshot

    @messages.handler
    def on_update(self, _: messages.UpdateEvent) -> None:
        geoms = self.viewer.extra_geoms
        geoms.clear()
        for marker in self.ui.markers:
            geoms.extend(_marker_geoms(marker))

    @messages.handler
    def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        self.front.frames += 1
        send = self.viewer.send_to_sim
        imgui.SetNextWindowPos(imgui.Vec2(10, 10), int(imgui.Cond.FirstUseEver))
        visible, _ = imgui.Begin(self.panel.title, flags=int(imgui.WindowFlags.AlwaysAutoResize))
        if visible:
            lines = iter(self.ui.lines)
            for i, element in enumerate(self.panel._elements):
                imgui.PushID(i)
                if isinstance(element, panels._Text):
                    imgui.TextUnformatted(next(lines, ""))
                elif isinstance(element, panels._Button):
                    if imgui.Button(element.label + (f"  ({element.key})" if element.key else "")):
                        send(_Input(i))
                    self._rect(i)
                elif isinstance(element, panels._Select):
                    changed, self.values[i] = imgui.Combo(element.label, self.values[i], element.options)
                    self._rect(i)
                    if changed:
                        send(_Input(i, self.values[i]))
                elif isinstance(element, panels._Slider):
                    changed, self.values[i] = imgui.SliderFloat(
                        element.label, self.values[i], element.low, element.high
                    )
                    self._rect(i)
                    if changed:
                        send(_Input(i, self.values[i]))
                imgui.PopID()
            imgui.Separator()
            imgui.TextUnformatted(self.ui.status)
            self.front.status = self.ui.status
        imgui.End()
        held: frozenset = frozenset()
        if not imgui.GetIO().WantCaptureKeyboard:
            for key in self.panel._keys:
                if imgui.IsKeyPressed(_imgui_key(key), repeat=False):
                    send(_Press(key))
            held = frozenset(k for k in self.panel._held if imgui.IsKeyDown(_imgui_key(k)))
            if "shift" in self.panel._held and imgui.IsKeyDown(imgui.Key.RightShift):
                held |= {"shift"}
        camera = self.viewer.camera
        send(_Keyboard(held, tuple(camera_view(camera.azimuth, camera.elevation))))

    def _rect(self, i: int) -> None:
        lo, hi = imgui.GetItemRectMin(), imgui.GetItemRectMax()
        self.front.rects[i] = (lo.x, lo.y, hi.x, hi.y)


class Studio:
    """Shows a panel and the station in Studio's web viewer, at ``http://127.0.0.1:port/``.

    ``port`` 0 takes the first free port from 8080. ``camera``: the initial view, (eye,
    look-at point) [m] in the station frame.
    """

    name = "studio"

    def __init__(self, panel: panels.Panel, *, port: int = 0, camera=None):
        self.panel = panel
        self.port = port
        self.camera = camera
        self.rects: dict[int, tuple[float, float, float, float]] = {}
        """Screen rectangles (x0, y0, x1, y1) [px] of the panel's buttons, lists and sliders by
        element index, as last drawn (for tests that click them)."""
        self.frames = 0
        """Frames drawn for a browser so far."""
        self.status = ""
        """The status line as last drawn."""
        self._ready = threading.Event()
        self._handle = None

    def open(self, station) -> str:
        """Start the viewer with the station's model; returns the page's address."""
        config = viewer_protocol.ViewerConfig(title=self.panel.title, http_port=self.port)
        with _localhost_only():
            self._handle = launch_web.launch(config, plugins=[_Sim(self.panel), _View(self)])
            # The first start in a process can take a minute (measured on a cloud Mac); wait as
            # long as Studio's thread lives. If it dies, is_running() raises its error.
            waited = 0.0
            while not self._ready.wait(0.1):
                waited += 0.1
                if not self._handle.is_running():
                    raise RuntimeError("MuJoCo Studio's viewer stopped while starting")
                if round(waited, 1) == 5.0:
                    print("Starting MuJoCo Studio; the first start can take a minute.", file=sys.stderr, flush=True)
        self._model, self._data = station.model, station.data
        self._handle.sync(self._model, self._data)  # sends the model; the page downloads it
        return f"http://127.0.0.1:{self.port}/"

    def is_open(self) -> bool:
        return self._handle is not None and self._handle.is_running()

    def poll(self) -> None:
        """Send the station's state, then run the callbacks of the input that arrived."""
        self._handle.sync(self._model, self._data)

    def show(self, lines: list[str], markers: list[panels.Marker], status: str) -> None:
        """Send the decision's text lines, markers and status line."""
        self._handle.send_to_viewer(_Ui(tuple(lines), tuple(markers), status))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
