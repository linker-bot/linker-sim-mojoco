"""The playground and the viewer-neutral parts of ``linker_sim.viewer``, without a browser.

The playground is driven by setting the keys its panel reports as held, as a viewer would,
with linker-lab's self-test script (``linker_lab.playground.SELFTEST``) and pass marks.
"""

import time

import numpy as np
import pytest

from linker_sim import hand, physics, playground, viewer, workstation
from linker_sim.viewer import Axes, Panel, Sphere

HZ = physics.DECISION_HZ


@pytest.fixture(scope="module")
def pg():
    return playground.Playground(playground.station())


def hold(pg, keys, seconds, then=0.0):
    """Hold ``keys`` for ``seconds``, then release them for ``then`` seconds."""
    pg.panel.held = frozenset(keys)
    for _ in range(round(seconds * HZ)):
        pg.step()
    pg.panel.held = frozenset()
    for _ in range(round(then * HZ)):
        pg.step()


def press(pg, key):
    """What a viewer does when a key is pressed."""
    pg.panel._keys[key]()


def settle_time(pg, side, limit=2.0):
    """Seconds until the palm point is within 2 mm and 1 degree of its goal (None: not in
    ``limit``)."""
    for k in range(round(limit * HZ)):
        distance, angle = pg.reach_error(side)
        if distance < 0.002 and angle < np.radians(1):
            return k / HZ
        pg.step()
    return None


def test_keys_move_the_goal_and_the_arm_follows(pg):
    """linker-lab's self-test: 1 s of L moves the right palm 10 cm to the camera's right,
    0.6 s of Shift+J turns it 17 degrees, 3 pinches (within 3 s), Tab and 0.5 s of U move the left palm
    5 cm up, R resets; every move settles within 2 s."""
    pg.reset()
    pg.panel.view = np.array(playground.CAMERA["lookat"]) - np.array(playground.CAMERA["eye"])
    start = {s: pg.palm(s) for s in workstation.SIDES}
    _, right, _ = playground.camera_axes(pg.panel.view)

    hold(pg, {"l"}, 1.0)
    assert settle_time(pg, "right") is not None
    moved = pg.palm("right")[0] - start["right"][0]
    assert np.linalg.norm(moved) > 0.09 and moved @ right > 0.09

    hold(pg, {"shift", "j"}, 0.6)
    assert settle_time(pg, "right") is not None
    turned = pg.palm("right")[1] @ start["right"][1].T
    assert np.degrees(np.arccos((np.trace(turned) - 1) / 2)) > 15

    press(pg, "3")
    hold(pg, set(), 3.0)  # the thumb yaw turns from home (1.36 rad) at about 2 rad/s
    shafts = pg.station.observe()[13 + 7 : 26]
    assert np.abs(shafts - hand.PRESETS["pinch"]).max() < 0.05

    press(pg, "tab")
    assert pg.active == "left"
    hold(pg, {"u"}, 0.5)
    assert settle_time(pg, "left") is not None
    assert pg.palm("left")[0][2] - start["left"][0][2] > 0.04

    assert pg.station.actions["arm_left"].limit_hits[0] == 0  # the palm moves stay within the limits
    press(pg, "r")
    home = np.array([workstation.HOME[s][n] for s in workstation.SIDES for n in workstation.ARM_JOINTS[s] + workstation.HAND_JOINTS[s]])
    positions = pg.station.observe()[:26]
    assert np.abs(positions - home).max() < 1e-3
    assert pg.active == "left"  # reset keeps the arm


def test_goal_stays_within_the_lead(pg):
    """Holding a key long moves the goal at most LEAD ahead of the palm point."""
    pg.reset()
    pg.active = "right"
    hold(pg, {"i"}, 0.3)
    distance, _ = pg.reach_error("right")
    assert distance <= playground.LEAD[0] + 1e-9


def test_drag_handle_moves_the_goal(pg):
    """The drag handle (viser) puts the active goal anywhere, beyond the lead."""
    pg.reset()
    pos, quat = pg.panel._handle.pose()
    pg.panel._handle.on_move(np.asarray(pos) + (0.0, 0.0, 0.08), quat)
    assert np.isclose(pg.reach_error(pg.active)[0], 0.08)
    assert settle_time(pg, pg.active) is not None


def test_panel_rejects_what_a_viewer_cannot_show():
    panel = Panel()
    panel.button("A", lambda: None, key="a")
    with pytest.raises(ValueError):
        panel.key("a", lambda: None)  # used twice
    with pytest.raises(ValueError):
        panel.hold("a")
    with pytest.raises(ValueError):
        panel.key("shift", lambda: None)
    with pytest.raises(ValueError):
        panel.key("F1", lambda: None)
    with pytest.raises(ValueError):
        panel.select("S", ["x", "x"], lambda i: None)
    with pytest.raises(ValueError):
        panel.slider("V", 0.0, 1.0, lambda v: None, value=2.0)


class FakeFront:
    """A front end without a viewer: records what the loop shows and feeds it input."""

    def __init__(self, panel, inputs):
        self.panel = panel
        self.inputs = inputs  # decision -> callable run as that decision's input
        self.shown = []
        self.decisions = 0

    def is_open(self):
        return True

    def poll(self):
        self.inputs.get(self.decisions, lambda: None)()
        self.decisions += 1

    def show(self, lines, markers, status):
        self.shown.append((lines, markers, status))


def test_loop_runs_input_step_and_drawing_in_order():
    """Each decision: the input's callbacks, then the step, then the text lines and markers
    made from the state after the step; the status line shows the speed."""
    station = workstation.Station(floor=False)
    panel = Panel("test")
    clicks = []
    panel.button("Click", lambda: clicks.append(station.time), key="c")
    panel.text(lambda: f"{station.time:.4f}")
    panel.markers(lambda: [Sphere((0, 0, 0)), Axes((0, 0, 0))])
    front = FakeFront(panel, {10: panel._keys["c"]})
    start = time.perf_counter()
    stats = viewer._loop(station, front, seconds=1.0)
    wall = time.perf_counter() - start
    assert stats.decisions == HZ and np.isclose(station.time, 1.0)
    assert clicks == [pytest.approx(10 / HZ)]  # before the 11th decision
    lines, markers, _ = front.shown[0]
    assert lines == [f"{1 / HZ:.4f}"] and [type(m) for m in markers] == [Sphere, Axes]
    assert front.shown[-1][2].startswith("Speed: ")
    # Paced: never faster than real time. (How close to real time a computer gets is
    # doctor's check and the development workstation's measurement, not a test's.)
    assert wall >= 1.0 - 1 / HZ and stats.speed < 1.1
