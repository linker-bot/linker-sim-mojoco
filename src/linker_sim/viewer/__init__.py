"""Show a station and a panel in a viewer: MuJoCo Studio (default) or viser.

    from linker_sim import viewer
    from linker_sim.viewer import Panel
    panel = Panel("Lesson 1")
    panel.button("Reset", station.reset, key="r")
    viewer.run(station, panel)                       # open http://127.0.0.1:8080/
    viewer.run(station, panel, viewer="viser")       # the same panel in viser

:func:`run` owns the loop: each decision it runs the callbacks of the input that arrived,
then ``step()`` (by default ``station.step()``), then draws, and it waits so that the
simulation runs at real time. Both viewers are web pages served from this computer
(``127.0.0.1``) and drawn by the browser; the simulation runs here, on the CPU.

Every panel ends with a status line: the speed (simulated seconds per second) and the
spare capacity (how much faster than real time the loop could run).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from linker_sim.viewer.panel import KEYS, Axes, Panel, Sphere, _Text

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["run", "Panel", "Sphere", "Axes", "KEYS", "Stats", "VIEWERS"]

VIEWERS = ("studio", "viser")


@dataclass
class Stats:
    """How a :func:`run` went."""

    decisions: int = 0
    """Decisions stepped."""
    speed: float = 0.0
    """Simulated seconds per wall-clock second over the last half second."""
    capacity: float = 0.0
    """Real time divided by the loop's own time per decision (without waiting): how much
    faster than real time it could run."""


def run(
    station,
    panel: Panel | None = None,
    *,
    step: Callable[[], None] | None = None,
    viewer: str = "studio",
    port: int = 0,
    camera=None,
    seconds: float | None = None,
) -> Stats:
    """Show ``station`` and ``panel`` in ``viewer`` until the page's server stops (Ctrl+C)
    or ``seconds`` of simulated time have passed.

    ``step()`` runs one decision; by default ``station.step()`` (every action repeats its
    previous command). Give your own to command the station, e.g.
    ``step=lambda: station.step(hand_left=command["hand"])``. ``port`` 0 takes the first free
    port from 8080. ``camera``: the initial view, (eye, look-at point) [m] in the station
    frame.
    """
    front = _front_end(viewer, panel or Panel(), port=port, camera=camera)
    try:
        url = front.open(station)
        print(f"{_NAMES[viewer]}: {url} (this computer only; Ctrl+C stops)", flush=True)
        return _loop(station, front, step=step, seconds=seconds)
    except KeyboardInterrupt:
        return Stats()
    finally:
        front.close()


_NAMES = {"studio": "MuJoCo Studio", "viser": "viser"}


def _front_end(viewer: str, panel: Panel, **options):
    if viewer == "studio":
        from linker_sim.viewer.studio import Studio

        return Studio(panel, **options)
    if viewer == "viser":
        from linker_sim.viewer.viser import Viser

        return Viser(panel, **options)
    raise ValueError(f"viewer must be one of {VIEWERS}, not {viewer!r}")


def _loop(station, front, *, step=None, seconds: float | None = None, realtime: bool = True) -> Stats:
    """The loop of :func:`run`, for an opened front end (tests use it directly)."""
    from linker_sim import physics

    panel = front.panel
    step = step or station.step
    dt = 1.0 / physics.DECISION_HZ
    stats = Stats()
    busy = 0.0  # loop time per decision without waiting, smoothed [s]
    clock = (time.perf_counter(), 0)  # (wall time, decisions) at the last speed reading
    paced = clock
    status = ""
    end = None if seconds is None else round(seconds * physics.DECISION_HZ)
    while front.is_open() and (end is None or stats.decisions < end):
        start = time.perf_counter()
        front.poll()
        step()
        stats.decisions += 1
        lines = [element.line() for element in panel._elements if isinstance(element, _Text)]
        markers = list(panel._draw()) if panel._draw is not None else []
        front.show(lines, markers, status)
        now = time.perf_counter()
        busy = 0.9 * busy + 0.1 * (now - start) if busy else now - start
        stats.capacity = dt / busy
        if now - clock[0] > 0.5:
            stats.speed = (stats.decisions - clock[1]) * dt / (now - clock[0])
            clock = (now, stats.decisions)
            status = f"Speed: {stats.speed:.2f}x real time (spare capacity {stats.capacity:.0f}x)"
        if realtime:  # never faster than real time
            ahead = (stats.decisions - paced[1]) * dt - (now - paced[0])
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.25:  # fell behind: don't catch up
                paced = (now, stats.decisions)
    return stats

