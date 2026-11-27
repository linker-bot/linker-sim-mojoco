"""The viewers in a browser: the viewer acceptance checks, for Studio
and viser.

Each test runs a station and a panel in one viewer and opens the page in headless Chrome
(``tests/browser.py``), as a user would: it waits for the scene to be drawn, clicks the
panel's buttons, lists and sliders, presses and holds keys, and checks that each reached
the simulation; that the markers are drawn; that the page also works embedded in an
``<iframe>`` of another page; that there are no browser console errors; and that the
panel shows the speed. The speed itself is printed, not checked: here a browser renders
LW-v0 in software on the same computer, so it measures that computer (``doctor`` checks
real time without a browser).

Skipped without Chrome. ``LINKER_SIM_VIEWER_SHOTS=DIR`` keeps screenshots there.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import browser
import numpy as np
import pytest
from PIL import Image

from linker_sim import gallery, hand, playground, viewer, workstation
from linker_sim.viewer import Axes, Panel, Sphere

pytestmark = pytest.mark.skipif(browser.chrome() is None, reason="needs Chrome (set CHROME)")

VIEWERS = tuple(os.environ.get("LINKER_SIM_BROWSER_VIEWERS", "studio viser").split())
"""The viewers these tests open: ``LINKER_SIM_BROWSER_VIEWERS`` (default both).

CI opens viser only. GitHub's runners have no GPU, so Chrome draws LW-v0 in software at
0.1-1.6 frames per second, and Studio reads input once per drawn frame: its keys took up
to 8 s to arrive and the tests failed by timing, not by a fault. The Studio tests run on
a computer with a GPU (hardware WebGL) before each merge; both viewers have not yet been
measured on computers with Intel or Apple GPUs. viser's panel and keys are page elements and stay
responsive under software rendering."""
PAGE_TIMEOUT = 120.0
"""Seconds the page may take until the scene is drawn (Studio downloads the whole model)."""
SHOTS = os.environ.get("LINKER_SIM_VIEWER_SHOTS")
IFRAME_TOP = 60
"""The test page's heading above the ``<iframe>`` [px]."""


INPUT_TIMEOUT = 30.0
"""Seconds an input may take to reach the simulation. Headless Chrome draws LW-v0 in
software, and Studio's input arrives at the pace the page draws: on the development
workstation a click on Studio's panel took about 1.7 s, a key 0.4 s; with Chrome and the
simulation on 2 cores a key took 4 s (and stayed held 4 s for a 1 s press). So every
test waits for the effect of one input before it sends the next."""


def eventually(b, condition, timeout=INPUT_TIMEOUT) -> bool:
    """Wait until ``condition()`` holds (False after ``timeout`` seconds)."""
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        b.wait(0.1)
    return bool(condition())


def front_end(name, panel, camera=None):
    return viewer._front_end(name, panel, port=browser.free_port(), camera=camera)


def pixels(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)


def scene_drawn(png: bytes) -> bool:
    """Whether the 3D scene is drawn: the middle of the window is not one colour."""
    middle = pixels(png)[300:800, 400:1100]
    background = np.median(middle.reshape(-1, 3), axis=0)
    return bool((np.abs(middle - background).sum(-1) > 30).mean() > 0.05)


def magenta(png: bytes) -> int:
    """Pixels of the test's magenta sphere."""
    a = pixels(png)
    return int(((a[..., 0] > 150) & (a[..., 2] > 150) & (a[..., 1] < 90)).sum())


def shot(b, name):
    png = b.screenshot()
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        Path(SHOTS, f"{name}.png").write_bytes(png)
    return png


def open_page(b, front, url, name):
    """Open ``url`` and wait until the scene is drawn; returns seconds until then."""
    start = time.time()
    b.open(url)
    while time.time() - start < PAGE_TIMEOUT:
        b.wait(1.0)
        if scene_drawn(b.screenshot()):
            break
    else:
        pytest.fail(f"{front.name}: no scene drawn within {PAGE_TIMEOUT:.0f} s")
    ready = time.time() - start
    # The scene can be drawn before the page drives the panel (Studio grants a page the panel
    # after it connects; a slow computer shows the gap): wait until the panel's frames run.
    if front.name == "studio":
        def frames_run():
            before = front.frames
            b.wait(0.5)
            return front.frames - before >= 10
        assert eventually(b, frames_run, timeout=PAGE_TIMEOUT), "studio: the page never drove the panel"
    else:
        assert eventually(b, lambda: front._server.get_clients(), timeout=PAGE_TIMEOUT)
    b.wait(1.0)
    # Focus the page: a click on the scene, then one key (Studio drops the first key after it).
    b.click(800, 600)
    b.press("x")
    b.wait(0.5)
    shot(b, f"{front.name}_{name}")
    return ready


def center(front, b, label):
    """Screen position [px] of the panel element labelled ``label``."""
    if front.name == "studio":
        index = next(i for i, e in enumerate(front.panel._elements) if getattr(e, "label", None) == label)
        x0, y0, x1, y1 = front.rects[index]
        return (x0 + x1) / 2, (y0 + y1) / 2
    # The panel's button shows the label, or the label and its key; viser's own hidden
    # buttons ("Reset View", ...) come first in the page.
    return b.evaluate(
        "(() => { const e = [...document.querySelectorAll('button')].find(b => b.getBoundingClientRect().width > 0"
        f" && (b.textContent === {label!r} || b.textContent.startsWith({label + '  ('!r}))); "
        "const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()"
    )


def hold(b, key, seconds, shift=False):
    """Hold ``key`` as a keyboard does: down, auto-repeat after 0.5 s at 30 Hz, up."""
    if shift:
        b.key("shift", True)
    b.key(key, True)
    start = time.time()
    b.wait(min(seconds, 0.5))
    text, code, vk = (key.upper() if shift else key), f"Key{key.upper()}", ord(key.upper())
    while time.time() - start < seconds:
        b.call("Input.dispatchKeyEvent", type="keyDown", key=text, code=code, windowsVirtualKeyCode=vk,
               text=text, autoRepeat=True, modifiers=8 if shift else 0)  # fmt: skip
        b.wait(1 / 30)
    b.key(key, False)
    if shift:
        b.key("shift", False)


def run_with_browser(station, front, script, step=None):
    """Open ``front``, run the loop here and ``script(browser, url)`` in a thread; returns the
    loop's stats and the script's result. The script's end stops the loop."""
    url = front.open(station)
    result, done = {}, threading.Event()

    def drive():
        b = None
        try:
            b = browser.Browser()
            result["value"] = script(b, url)
            result["errors"] = b.errors()
        except BaseException as error:  # re-raised below, on the test's thread
            result["error"] = error
        finally:
            try:
                if b is not None:
                    b.close()
            finally:
                done.set()

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    stats = viewer.Stats()

    class Until:  # the loop runs until the script is done
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def is_open(self):
            return not done.is_set() and self.inner.is_open()

    try:
        stats = viewer._loop(station, Until(front), step=step)
    finally:
        thread.join(30)
        front.close()
    if "error" in result:
        raise result["error"]
    return stats, result["value"], result["errors"]


@pytest.mark.parametrize("name", VIEWERS)
def test_panel_elements_reach_the_simulation(name):
    """Every panel element works in the page and in an ``<iframe>``: button (click and its
    key), list, slider, held keys (and Shift), text lines, markers, the status line; no
    console errors; real time."""
    station = workstation.Station()
    seen = {"clicks": 0, "picked": None, "level": None, "held": set()}
    panel = Panel("Test panel")
    panel.text(lambda: f"Clicks: {seen['clicks']}")
    panel.button("Count", lambda: seen.update(clicks=seen["clicks"] + 1), key="c")
    panel.select("Pick", ["one", "two", "three"], lambda i: seen.update(picked=i))
    panel.slider("Level", 0.0, 1.0, lambda v: seen.update(level=v), value=0.5)
    panel.hold("h", "shift")
    panel.markers(lambda: [Sphere((0.45, 0.0, -0.4), 0.06, (1.0, 0.0, 1.0, 1.0)), Axes((0.45, 0.0, -0.4))])
    front = front_end(name, panel, camera=(playground.CAMERA["eye"], playground.CAMERA["lookat"]))

    def step():
        station.step()
        seen["held"] |= panel.held

    def script(b, url):
        report = {"page_ready_s": open_page(b, front, url, "elements")}
        report["magenta_px"] = magenta(b.screenshot())
        clicked = time.time()
        b.click(*center(front, b, "Count"))
        eventually(b, lambda: seen["clicks"] == 1)
        report["click_lag_s"] = time.time() - clicked
        report["clicks_after_click"] = seen["clicks"]
        b.press("c")
        eventually(b, lambda: seen["clicks"] == 2)
        report["clicks_after_key"] = seen["clicks"]
        hold(b, "h", 1.0)
        eventually(b, lambda: "h" in seen["held"])
        eventually(b, lambda: not panel.held)
        report["held"] = set(seen["held"])
        seen["held"] = set()
        hold(b, "h", 1.0, shift=True)
        eventually(b, lambda: {"h", "shift"} <= seen["held"])
        eventually(b, lambda: not panel.held)
        report["held_shift"] = set(seen["held"])
        pick(b, front, "Pick", "two", lambda: seen["picked"] is not None, report["click_lag_s"])
        report["picked"] = seen["picked"]
        slide(b, front, "Level", lambda: seen["level"] is not None)
        report["level"] = seen["level"]
        report["status"] = status(b, front)
        # The same page in an <iframe> of a page from another origin, in a new browser: Studio's
        # page isolates itself from other origins (the first tab can't navigate there), and
        # gives the panel to one browser at a time (the others watch). The viewer's panel sits
        # at the same place in the frame, below the page's heading.
        x, y = center(front, b, "Count")
        b.close()
        page, server = browser.serve_page(
            f'<html><body style="margin:0"><h3 style="margin:0;height:{IFRAME_TOP}px">Embedding test</h3>'
            f'<iframe src="{url}" style="border:0;width:100vw;height:calc(100vh - {IFRAME_TOP}px)"></iframe>'
            "</body></html>"
        )
        embedding = browser.Browser()
        try:
            open_page(embedding, front, page, "iframe")
            embedding.click(x, y + IFRAME_TOP)
            eventually(embedding, lambda: seen["clicks"] == 3)
            report["clicks_in_iframe"] = seen["clicks"]
            report["iframe_errors"] = embedding.errors()
        finally:
            embedding.close()
            server.shutdown()
        return report

    stats, report, errors = run_with_browser(station, front, script, step=step)
    print(f"\n{name}: {report}, speed {stats.speed:.2f}x, spare capacity {stats.capacity:.1f}x")
    assert not errors and not report["iframe_errors"], (errors, report["iframe_errors"])
    assert report["magenta_px"] > 500  # the sphere: about 60 px across
    assert report["clicks_after_click"] == 1 and report["clicks_after_key"] == 2
    assert "h" in report["held"] and "shift" not in report["held"]
    assert {"h", "shift"} <= report["held_shift"]
    assert report["picked"] == 1
    assert report["level"] is not None and report["level"] != 0.5 and 0.0 <= report["level"] <= 1.0
    assert report["status"].startswith("Speed: ")
    assert report["clicks_in_iframe"] == 3


def pick(b, front, label, option, chosen, lag):
    """Choose ``option`` in the list ``label``; wait until ``chosen()``. ``lag``: seconds a
    click took to reach the simulation (Studio: the list must be open before its option is
    clicked, and nothing reaches the simulation until then)."""
    if front.name == "studio":
        index = next(i for i, e in enumerate(front.panel._elements) if getattr(e, "label", None) == label)
        x0, y0, x1, y1 = front.rects[index]
        b.click(x0 + 0.3 * (x1 - x0), (y0 + y1) / 2)
        b.wait(2 * lag + 2.0)
        row = y1 - y0  # the open list's rows are about as high as the box
        n = front.panel._elements[index].options.index(option)
        b.click(x0 + 0.3 * (x1 - x0), y1 + 0.5 * row + n * (row - 2) + 4)
    else:
        b.evaluate(
            "(() => { const l = [...document.querySelectorAll('label')].find(l => l.textContent === "
            f"{label!r}); document.getElementById(l.htmlFor).click(); }})()"
        )
        b.wait(0.5)
        x, y = b.evaluate(
            "(() => { const o = [...document.querySelectorAll('[role=option]')].find(o => o.textContent === "
            f"{option!r}); const r = o.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"
        )
        b.click(x, y)
    eventually(b, chosen)


def slide(b, front, label, changed):
    """Click into the slider ``label``, away from its start value; wait until ``changed()``."""
    if front.name == "studio":
        index = next(i for i, e in enumerate(front.panel._elements) if getattr(e, "label", None) == label)
        x0, y0, x1, y1 = front.rects[index]
        b.click(x0 + 0.2 * (x1 - x0), (y0 + y1) / 2)
    else:
        x, y = b.evaluate(
            "(() => { const s = document.querySelector('[role=slider]').parentElement.getBoundingClientRect();"
            " return [s.x + 0.8 * s.width, s.y + s.height / 2]; })()"
        )
        b.click(x, y)
    eventually(b, changed)


def status(b, front) -> str:
    """The panel's status line as shown."""
    if front.name == "studio":
        return front.status
    return b.evaluate("[...document.querySelectorAll('p')].map(p => p.textContent).find(t => t.startsWith('Speed:')) || ''")


@pytest.mark.parametrize("name", VIEWERS)
def test_playground_in_the_browser(name):
    """The playground through the page: holding L moves the right palm point to the
    camera's right, Shift+J turns it, the Pinch button pinches, Tab switches arm, R resets;
    at real time, with no console errors."""
    pg = playground.Playground(playground.station())
    front = front_end(name, pg.panel, camera=(playground.CAMERA["eye"], playground.CAMERA["lookat"]))
    start = {s: pg.palm(s) for s in workstation.SIDES}

    def script(b, url):
        report = {"page_ready_s": open_page(b, front, url, "playground")}
        _, right, _ = playground.camera_axes(pg.panel.view)
        moved = lambda: float((pg.palm("right")[0] - start["right"][0]) @ right)  # noqa: E731

        def turned():
            r = pg.palm("right")[1] @ start["right"][1].T
            return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))

        def settled():
            """The keys are released, and the palm point has reached its goal."""
            goal = pg.goal["right"][0].copy()
            b.wait(0.5)
            still = not pg.panel.held and np.allclose(goal, pg.goal["right"][0])
            return still and pg.reach_error("right")[0] < 0.002

        hold(b, "l", 1.0)
        eventually(b, lambda: moved() > 0.02)
        eventually(b, settled)
        report["moved_right_m"] = moved()
        hold(b, "j", 1.0, shift=True)  # viser: one tap, then the keyboard's repeat from 0.5 s
        eventually(b, lambda: turned() > 5)
        eventually(b, settled)
        report["turned_deg"] = turned()
        b.click(*center(front, b, "Pinch"))
        pinch = lambda: float(np.abs(pg.station.observe()[20:26] - hand.PRESETS["pinch"]).max())  # noqa: E731
        eventually(b, lambda: pinch() < 0.05)  # about 3 s after the click arrives
        report["pinch_error_rad"] = pinch()
        shot(b, f"{front.name}_playground_moved")
        b.press("tab")
        eventually(b, lambda: pg.active == "left")
        report["active"] = pg.active
        steps = pg.station.time
        b.press("r")
        eventually(b, lambda: pg.station.time < steps)
        b.wait(0.5)
        home = np.array([workstation.HOME["right"][n] for n in workstation.ARM_JOINTS["right"]])
        report["home_error_rad"] = float(np.abs(pg.station.observe()[13:20] - home).max())
        report["status"] = status(b, front)
        return report

    stats, report, errors = run_with_browser(pg.station, front, script, step=pg.step)
    print(f"\n{name}: {report}, speed {stats.speed:.2f}x, spare capacity {stats.capacity:.1f}x")
    assert not errors, errors
    # 1 s held: Studio jogs 1 s at 0.1 m/s (at most LEAD, 5 cm, ahead of the palm); viser
    # moves 1 cm per tap and its count depends on the keyboard's repeat reaching the page.
    assert report["moved_right_m"] > 0.02
    assert report["turned_deg"] > 5  # 0.5 rad/s, at most LEAD (17 degrees) ahead
    assert report["pinch_error_rad"] < 0.05
    assert report["active"] == "left"
    assert report["home_error_rad"] < 1e-3
    assert report["status"].startswith("Speed: ")


FILAMENT_ARENA = "HandleAllocator arena is full, using slower system heap"
"""A notice Studio's renderer (Filament, in the page) logs as an error when a model has more
meshes than its fixed handle arena holds: ycb-grasp 0.2.0's 38 objects brought 1,060 meshes (their
collision pieces and one visual mesh each). The renderer then allocates from the browser's
heap; the page still draws and runs (measured). The arena's size
is compiled into Studio."""


def page_errors(errors: list[str]) -> list[str]:
    """Console errors, without Filament's arena notice (a panic log and its empty line)."""
    return [e for e in errors if FILAMENT_ARENA not in e and e.strip()]


@pytest.mark.parametrize("name", VIEWERS)
def test_gallery_in_the_browser(name):
    """The ycb-grasp entry through the page: choose the tomato soup can in the list, Drop
    (it comes to rest), Grasp (the episode ends in "grasped"), Reset (the table is cleared);
    at real time, with no console errors (but :data:`FILAMENT_ARENA` in Studio)."""
    play = gallery.load("ycb-grasp")
    scene = play.scene(gallery.fetch("ycb-grasp"))
    scene.rng = np.random.default_rng(5)  # tests/test_gallery.py's placement: mid-region (x 0.42, y 0.16)
    front = front_end(name, play.make_panel(scene), camera=(play.CAMERA["eye"], play.CAMERA["lookat"]))
    soup = next(i for i, o in enumerate(scene.items) if o["name"] == "005_tomato_soup_can")

    def script(b, url):
        report = {"page_ready_s": open_page(b, front, url, "gallery")}
        clicked, before = time.time(), scene.station.time
        b.click(*center(front, b, "Reset"))  # a click's lag: Reset sets the station's time to 0
        eventually(b, lambda: scene.station.time < before)
        report["click_lag_s"] = time.time() - clicked
        pick(b, front, "Object", scene.items[soup]["label"], lambda: scene.choice == soup, report["click_lag_s"])
        report["chosen"] = scene.choice == soup
        b.click(*center(front, b, "Drop"))
        report["at_rest"] = eventually(b, lambda: scene.active == soup and scene.state == "ready")
        shot(b, f"{front.name}_gallery_ready")
        b.click(*center(front, b, "Grasp"))
        eventually(b, lambda: scene.state == "grasping")
        report["done"] = eventually(b, lambda: scene.state == "done", timeout=INPUT_TIMEOUT + 20)
        report["message"] = scene.message
        shot(b, f"{front.name}_gallery_done")
        b.click(*center(front, b, "Reset"))
        eventually(b, lambda: scene.state == "empty")
        report["state_after_reset"] = scene.state
        report["status"] = status(b, front)
        return report

    stats, report, errors = run_with_browser(scene.station, front, script, step=scene.step)
    print(f"\n{name}: {report}, speed {stats.speed:.2f}x, spare capacity {stats.capacity:.1f}x")
    assert not page_errors(errors), errors
    assert report["chosen"] and report["at_rest"] and report["done"], report
    assert report["message"] == "Tomato soup can: grasped.", report["message"]
    assert report["state_after_reset"] == "empty"
    assert report["status"].startswith("Speed: ")


@pytest.mark.parametrize("name", VIEWERS)
def test_lesson_in_the_browser(name):
    """The worked lesson (``examples/pick_up_a_box.py``) through the page: key 1 drops the box,
    it comes to rest, key 2 grasps it; the policy lifts it more than 8 cm above the table
    and holds it ("grasped"), with no console errors."""
    from test_examples import load

    lesson = load("pick_up_a_box")
    lesson.scene.rng = np.random.default_rng(0)
    front = front_end(name, lesson.panel)
    scene = lesson.scene

    def script(b, url):
        report = {"page_ready_s": open_page(b, front, url, "lesson")}
        b.press("1")
        report["dropped"] = eventually(b, lambda: scene.state == "ready")
        b.press("2")
        report["lifted"] = eventually(b, lambda: lesson.box_height() > 0.08)
        shot(b, f"{front.name}_lesson_lifted")
        report["grasped"] = eventually(b, lambda: scene.message.endswith(": grasped."))
        report["height_m"] = round(float(lesson.box_height()), 3)
        return report

    stats, report, errors = run_with_browser(lesson.station, front, script, step=lesson.step)
    print(f"\n{name}: {report}, speed {stats.speed:.2f}x, spare capacity {stats.capacity:.1f}x")
    assert not page_errors(errors), errors
    assert all(report[k] for k in ("dropped", "lifted", "grasped")), report


@pytest.mark.skipif("viser" not in VIEWERS, reason="LINKER_SIM_BROWSER_VIEWERS leaves out viser")
def test_drag_handle_in_viser():
    """Dragging the playground's goal handle in viser (its red arrow, the palm normal,
    pointing down) moves the goal down, and the arm follows. Studio has no handle."""
    import viser.transforms

    pg = playground.Playground(playground.station())
    front = front_end("viser", pg.panel, camera=(playground.CAMERA["eye"], playground.CAMERA["lookat"]))

    def project(point):
        """Screen position [px] of a point, through the browser's camera (OpenCV axes)."""
        camera = next(iter(front._server.get_clients().values())).camera
        rotation = viser.transforms.SO3(np.asarray(camera.wxyz)).as_matrix()
        x, y, z = rotation.T @ (np.asarray(point) - np.asarray(camera.position))
        f = 0.5 * camera.image_height / np.tan(camera.fov / 2)
        return 0.5 * camera.image_width + f * x / z, 0.5 * camera.image_height + f * y / z

    def script(b, url):
        open_page(b, front, url, "drag")
        start, rotation = pg.goal["right"]
        start = start.copy()
        x, y = project(start + 0.6 * 0.15 * rotation[:, 0])  # on the red arrow (handle scale 0.15 m)
        b.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="none", buttons=0)
        b.wait(0.3)
        b.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", buttons=1, clickCount=1)
        b.wait(0.2)
        for k in range(1, 21):  # 60 px down, in 20 moves
            b.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y + 3 * k, button="left", buttons=1)
            b.wait(0.03)
        b.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y + 60, button="left", buttons=0, clickCount=1)
        eventually(b, lambda: pg.goal["right"][0][2] < start[2] - 0.05)
        moved = pg.goal["right"][0] - start
        settled = eventually(b, lambda: pg.reach_error("right")[0] < 0.002)
        shot(b, "viser_dragged")
        return {"goal_moved_m": moved, "settled": settled, "dragging": front._dragging}

    stats, report, errors = run_with_browser(pg.station, front, script, step=pg.step)
    print(f"\nviser drag: {report}, speed {stats.speed:.2f}x")
    assert not errors, errors
    assert report["goal_moved_m"][2] < -0.05 and np.abs(report["goal_moved_m"][:2]).max() < 0.01
    assert report["settled"] and not report["dragging"]
