"""The worked lesson in ``examples/`` runs and does what ``docs/lesson.md`` says."""

import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str):
    """An example script as a module (``examples/`` is not a package); each load is a fresh
    station."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lesson_lifts_the_box():
    """The lesson without a viewer: the box is dropped, comes to rest, and the policy grasps it
    (lifted 10 cm and held 2 s, touching nothing but the hand)."""
    assert load("pick_up_a_box").main(["--check"]) == 0


def test_lesson_panel():
    """The lesson's panel: three text lines, Drop, Grasp and Reset on keys 1-3."""
    lesson = load("pick_up_a_box")
    panel = lesson.panel
    assert set(panel._keys) == {"1", "2", "3"}
    lines = [e.line() for e in panel._elements if hasattr(e, "line")]
    assert lines == ["Pick an object and press Drop.", "Palm to box: -", "Box above the table: 0.0 cm"]
    panel._keys["1"]()
    lesson.step()
    assert lesson.scene.state == "settling" and lesson.box_height() < 0.01


def test_lesson_page_is_the_example():
    """``docs/lesson.md`` builds the example: each of its code blocks that defines the lesson
    (not the ones that try it out or run the viewer) appears in the example as it is."""
    import re

    def lines(text):
        return "\n".join(line for line in text.splitlines() if line.strip() and not re.match(r"# \d\. ", line))

    page = (EXAMPLES.parent / "docs" / "lesson.md").read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", page, flags=re.S | re.M)
    defining = [b for b in blocks if not b.startswith(("for ", "viewer.run"))]
    assert len(defining) == 4
    example = lines((EXAMPLES / "pick_up_a_box.py").read_text())
    for block in defining:
        assert lines(block) in example, block.splitlines()[0]
