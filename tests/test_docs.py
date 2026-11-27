"""The user docs are the public API's contract: they list exactly what each public module
exports, every documented member exists, and their examples run.

- ``docs/api.md`` names each public module in a ``## `linker_sim.<module>``` heading and
  each public name in a ``### `<name>...``` heading below it; the module's ``__all__`` must
  be the same names.
- A class section's top-level bullets (``- `member...`: ...``) list its members; they must be
  exactly the class's public members (an instance's, for the classes made by a station).
- Every ``python`` code block of the README and the docs runs, one file at a time in one
  namespace, with ``viewer.run`` replaced by a stand-in that only records its call.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api.md"


def sections() -> dict[str, dict[str, list[str]]]:
    """Module -> public name -> the members its section's top-level bullets name."""
    modules: dict[str, dict[str, list[str]]] = {}
    name = None
    # A bullet's continuation lines (indented, not a nested bullet) join its first line.
    text = re.sub(r"\n  (?![ -])", " ", API.read_text())
    for line in text.splitlines():
        if m := re.match(r"## `linker_sim\.(\w+)`", line):
            module = modules.setdefault(m[1], {})
            name = None
        elif m := re.match(r"### `(\w+)", line):
            name = m[1]
            module[name] = []
        elif line.startswith("## "):
            name = None
        elif name and line.startswith("- `"):
            head = line[2:].split("`: ", 1)[0] + "`"  # the backticked names before the colon
            module[name] += [re.match(r"\w+", group)[0] for group in re.findall(r"`([^`]+)`", head)]
    return modules


def public(names) -> set[str]:
    return {n for n in names if not n.startswith("_")}


@pytest.mark.parametrize("module", sorted(sections()))
def test_public_names_are_the_documented_ones(module):
    documented = set(sections()[module])
    exported = set(importlib.import_module(f"linker_sim.{module}").__all__)
    assert exported == documented, (
        f"linker_sim.{module}: in __all__ only {sorted(exported - documented)}, "
        f"in docs/api.md only {sorted(documented - exported)}"
    )


@pytest.fixture(scope="module")
def station():
    from linker_sim import objects
    from linker_sim.workstation import Station

    box = objects.object_cfg([([(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, 0, 0.05)], [(0, 1, 2), (0, 1, 3),
                              (0, 2, 3), (1, 2, 3)])], 0.1)  # fmt: skip
    return Station(objects={"box": box})


def members(module: str, name: str, station) -> set[str]:
    """The public members of a documented class: an instance's where the station makes it."""
    from linker_sim import viewer

    instances = {
        "Station": lambda: station,
        "RigidObject": lambda: station.objects["box"],
        "ArmAction": lambda: station.actions["arm_left"],
        "HandAction": lambda: station.actions["hand_left"],
        "Panel": viewer.Panel,
    }
    cls = getattr(importlib.import_module(f"linker_sim.{module}"), name)
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return public(dir(instances[name]())) - public(dir(object))


def test_documented_members_are_the_public_ones(station):
    checked = []
    for module, names in sections().items():
        for name, documented in names.items():
            obj = getattr(importlib.import_module(f"linker_sim.{module}"), name)
            if documented and isinstance(obj, type):  # a class section with members (a function's are its arguments)
                assert set(documented) == members(module, name, station), f"{module}.{name}"
                checked.append(name)
    assert {"Station", "RigidObject", "ArmAction", "HandAction", "Panel"} <= set(checked)


DOCS = sorted([ROOT / "README.md", *ROOT.glob("docs/**/*.md")])


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_in_the_docs_run(doc, monkeypatch):
    from linker_sim import viewer

    blocks = re.findall(r"^```python\n(.*?)^```", doc.read_text(), flags=re.S | re.M)
    calls = []
    monkeypatch.setattr(viewer, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace: dict = {"__name__": "docs_example"}
    for block in blocks:
        exec(compile(block, f"{doc.name} example", "exec"), namespace)  # noqa: S102
    for (station, *_), kwargs in calls:  # what was shown steps: one decision of it runs
        (kwargs.get("step") or station.step)()
