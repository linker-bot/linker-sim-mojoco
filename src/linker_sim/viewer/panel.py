"""A panel: an interface for a station, described once and shown by either viewer.

A :class:`Panel` lists what the interface offers: lines of text, buttons (each optionally
on a key), lists to choose from, sliders, keys, held keys, markers drawn in the scene and
a drag handle for a pose. It imports no viewer; ``linker_sim.viewer.run`` shows it in
MuJoCo Studio or in viser:

    panel = Panel("Lesson 1")
    panel.text(lambda: f"Box height: {station.objects['box'].pose()[0][2]:+.3f} m")
    panel.button("Close the hand", lambda: command.update(hand=hand.PRESETS["close"]), key="c")
    panel.markers(lambda: [Sphere(goal, radius=0.01)])

Every callback (``on_click``, ``on_select``, ``on_change``, ``on_press``, ``on_move`` and
the functions that make text lines and markers) runs on the thread that runs the station,
between two decisions, so it may read and command the station directly. Text lines and
markers are made again after every decision.

The two viewers differ in two things (``docs/viewers.md``):

- **Held keys** (:meth:`Panel.hold`): Studio reports how long a key is held. viser
  reports presses only, so there each press counts as the key held for a short tap
  (``linker_sim.viewer.viser.TAP``, 0.1 s), extended by the keyboard's auto-repeat.
- **The drag handle** (:meth:`Panel.handle`) is drawn by viser only; Studio has no
  transform handle, so offer the same pose through keys or sliders too.

Positions are in the station frame [m], quaternions x, y, z, w (as everywhere in
linker-sim).
"""

from __future__ import annotations

import string
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

KEYS = frozenset((*string.ascii_lowercase, *string.digits, "tab", "space", "shift"))
"""The keys a panel can use: letters, digits, Tab, Space, and Shift (held only).

viser moves its camera with W, A, S, D, Q, E and the arrow keys, so a panel meant for
both viewers leaves those letters free."""


@dataclass(frozen=True)
class Sphere:
    """A sphere at ``pos`` [m], ``radius`` [m], colour ``rgba`` (each 0..1)."""

    pos: Sequence[float]
    radius: float = 0.01
    rgba: Sequence[float] = (0.1, 0.8, 0.2, 0.6)


@dataclass(frozen=True)
class Axes:
    """The axes of a frame at a pose: x red, y green, z blue, each ``length`` [m] long."""

    pos: Sequence[float]
    quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)
    length: float = 0.1


Marker = Sphere | Axes


@dataclass
class _Text:
    line: Callable[[], str]


@dataclass
class _Button:
    label: str
    on_click: Callable[[], None]
    key: str | None = None


@dataclass
class _Select:
    label: str
    options: tuple[str, ...]
    on_select: Callable[[int], None]
    index: int = 0


@dataclass
class _Slider:
    label: str
    low: float
    high: float
    on_change: Callable[[float], None]
    value: float = 0.0


@dataclass
class _Handle:
    pose: Callable[[], tuple[Sequence[float], Sequence[float]]]
    on_move: Callable[[np.ndarray, np.ndarray], None]


_Element = _Text | _Button | _Select | _Slider


class Panel:
    """One interface for a station: a titled panel of elements, shown in the order they
    are added, plus keys, held keys, markers and a drag handle.

    Build it before ``linker_sim.viewer.run`` shows it: the elements are fixed while it
    runs (the text of a line, the markers and the handle's pose are not).
    """

    def __init__(self, title: str = "linker-sim"):
        self.title = title
        self.held: frozenset[str] = frozenset()
        """The keys of :meth:`hold` that are held down now, updated before every decision."""
        self.view = np.array([1.0, 0.0, 0.0])
        """The direction the viewer's camera looks in (unit vector, station frame), updated
        before every decision; e.g. for keys that move a goal relative to the view."""
        self._elements: list[_Element] = []
        self._keys: dict[str, Callable[[], None]] = {}
        self._held: frozenset[str] = frozenset()
        self._draw: Callable[[], Iterable[Marker]] | None = None
        self._handle: _Handle | None = None

    def text(self, line: Callable[[], str]) -> None:
        """A line of text: ``line()`` is called after every decision and shown."""
        self._elements.append(_Text(line))

    def button(self, label: str, on_click: Callable[[], None], *, key: str | None = None) -> None:
        """A button; ``key`` (one of :data:`KEYS`) presses it from the keyboard too."""
        if key is not None:
            self.key(key, on_click)
        self._elements.append(_Button(label, on_click, key))

    def select(
        self, label: str, options: Sequence[str], on_select: Callable[[int], None], *, index: int = 0
    ) -> None:
        """A list to choose one of ``options`` from (a drop-down list), ``index`` chosen at the
        start. Choosing calls ``on_select(i)`` with the chosen index."""
        options = tuple(options)
        if len(set(options)) != len(options):
            raise ValueError("the options must differ from each other")
        if not 0 <= index < len(options):
            raise ValueError(f"index {index} is not one of the {len(options)} options")
        self._elements.append(_Select(label, options, on_select, index))

    def slider(
        self, label: str, low: float, high: float, on_change: Callable[[float], None], *, value: float | None = None
    ) -> None:
        """A slider from ``low`` to ``high``, starting at ``value`` (default ``low``). Moving it
        calls ``on_change(v)`` with the new value."""
        value = low if value is None else value
        if not low <= value <= high:
            raise ValueError(f"value {value} is not within {low}..{high}")
        self._elements.append(_Slider(label, float(low), float(high), on_change, float(value)))

    def key(self, key: str, on_press: Callable[[], None]) -> None:
        """Call ``on_press()`` when ``key`` (one of :data:`KEYS`, not Shift) is pressed."""
        if key not in KEYS or key == "shift":
            raise ValueError(f"key must be a letter, a digit, 'tab' or 'space', not {key!r}")
        if key in self._keys or key in self._held:
            raise ValueError(f"key {key!r} is used twice")
        self._keys[key] = on_press

    def hold(self, *keys: str) -> None:
        """Report whether ``keys`` (of :data:`KEYS`) are held down, in :attr:`held`; e.g. to
        move something while a key is held. Include ``"shift"`` to read Shift with them."""
        for key in keys:
            if key not in KEYS:
                raise ValueError(f"{key!r} is not one of the keys a panel can use")
            if key in self._keys:
                raise ValueError(f"key {key!r} is used twice")
        self._held |= frozenset(keys)

    def markers(self, draw: Callable[[], Iterable[Marker]]) -> None:
        """Markers drawn in the scene: ``draw()`` returns them (:class:`Sphere`, :class:`Axes`)
        and is called after every decision."""
        self._draw = draw

    def handle(
        self,
        pose: Callable[[], tuple[Sequence[float], Sequence[float]]],
        on_move: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """A drag handle for a pose (viser only): drawn at ``pose()`` (position, quaternion),
        which is read after every decision; dragging it calls ``on_move(pos, quat)``."""
        self._handle = _Handle(pose, on_move)
