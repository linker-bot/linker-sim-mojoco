"""Check that the code copied from linker-lab is unchanged: check_copies.py LINKER_LAB_CHECKOUT.

linker-lab is the source of the engine-free code (the arm and hand models, the reading of
LW-v0's files and their physics values, the object maths, the playground's jog, the
gallery's runner and the ``ycb-grasp`` entry's policy, kinematics and episode, their
tests). linker-sim copies it from a linker-lab release, the tag :data:`COMMIT`; a fix is made
in linker-lab first, released, then copied. This script reads each copied definition from
linker-lab at that tag (``git show``, so the checkout's working tree may be anywhere) and compares it with linker-sim's: the same syntax tree, so comments
and formatting may differ, code and docstrings may not. ``linker_lab`` is read as
``linker_sim``.

It also checks that each copied file names the release it was copied from. Exit status 0
when every copy matches. Opt-in (it needs a linker-lab checkout): ``tests/test_copies.py``
runs it when ``LINKER_LAB_CHECKOUT`` is set; part of the release gate.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

COMMIT = "v1.0.0"
"""The linker-lab release (its tag) every copy is taken from."""

ROOT = Path(__file__).resolve().parents[1]

COPIES = {
    "src/linker_sim/arm.py": (
        "src/linker_lab/arm.py",
        "_CONTROLLER VELOCITY ACCELERATION JERK WRIST_OCTAGON TORQUE_GUARD_MARGIN F64 _vec7 _BISECTIONS _Profile "
        "_advance _velocity_change _profile _state_at _duration _travel _fastest _within _fastest_kernel "
        "_sample_kernel plan feed_forward _guard_kernel torque_guard hermite _project_kernel project "
        "_rate_limit_kernel rate_limit _check",
    ),
    "src/linker_sim/hand.py": ("src/linker_lab/hand.py", "SHAFTS PRESETS THUMB_OPPOSE_YAW thumb_oppose HandMotor"),
    "src/linker_sim/assets.py": (
        "src/linker_lab/assets.py",
        "NAME PARTS root path part_file cache_dir sha256 scene part_pose _model home camera MOTOR_MODEL _data _unit_joints motor controller",
    ),
    "src/linker_sim/physics.py": ("src/linker_lab/physics.py", "_option _size _flag PHYSICS_HZ DECISION_HZ DECIMATION SOLVER CAPACITY"),
    "src/linker_sim/objects.py": (
        "src/linker_lab/objects.py",
        "Part MATERIAL load_mesh mass_properties table on_table _rotation",
    ),
    "src/linker_sim/playground.py": (
        "src/linker_lab/playground.py",
        "JOG_SPEED LEAD BOX CAMERA KEYS ACTIONS camera_axes jog",
    ),
    "src/linker_sim/gallery.py": (
        "src/linker_lab/gallery.py",
        "ROOT FIELDS entries manifest cache_dir artifact_path fetch load",
    ),
    "policy-gallery/ycb-grasp/play.py": (
        "policy-gallery/ycb-grasp/play.py",
        "STATE_DIM POINTS ACTION_SCALE DROP_CLEARANCE SETTLE_TICKS QUIET_TICKS WINDOW RETRIES "
        "LIFT_HEIGHT RISE IK_ITERATIONS PLAN_POSITION_TOLERANCE PLAN_ROTATION_TOLERANCE PLAN_MAX_STEP "
        "CONTACT_TOLERANCE JOINT_LIMIT_TOLERANCE CAMERA load_objects label load_policy PointNet Actor LeftArm "
        "_rotation _rotation_error _smoothstep parking merged GraspScene.press GraspScene._drop "
        "GraspScene._grasp GraspScene._settle_check GraspScene._grasp_targets GraspScene.plan "
        "GraspScene.plan_lift",
    ),
    "tests/test_drive_reference.py": ("tests/test_drive_reference.py", "*"),
    "tests/test_arm.py": ("tests/test_arm.py", "*"),
    "tests/test_hand.py": ("tests/test_hand.py", "*"),
    "tests/test_assets.py": (
        "tests/test_assets.py",
        "test_files_come_from_the_package test_the_installed_files_are_the_pinned_release "
        "test_motor_constants_and_derived_passive_damping test_arm_controller_limits",
    ),
    "tests/test_lw_v0.py": ("tests/test_lw_v0.py", "*"),
    "tests/test_objects.py": (
        "tests/test_objects.py",
        "box test_box_mass_properties test_two_parts_add_up_like_one_body test_load_mesh test_table_and_placement",
    ),
}
"""linker-sim file -> (linker-lab file, the copied names: top-level, or ``Class.method``;
``*``: every top-level one)."""

CONSTANTS = {
    # HandActionCfg's option default is copied; the class around it is new (the motor
    # constants are linker-sim-assets' hands/o6/motor.yaml, read by assets.motor).
    "src/linker_sim/hand.py": ("src/linker_lab/hand.py", "HandActionCfg", "normalized"),
    "src/linker_sim/arm.py": (
        "src/linker_lab/arm.py",
        "ArmActionCfg",
        "command interpolation stream limits body_name body_offset ik_damping",
    ),
}
"""Fields of a config class whose defaults are copied: file -> (linker-lab file, class, fields)."""

DATA = {"tests/data/drive_reference.npz": "tests/data/drive_reference.npz"}
"""Copied data files, compared byte for byte."""


def definitions(source: str) -> dict[str, ast.AST]:
    """Top-level functions, classes and assignments of a module, by name, and the methods
    of its classes as ``Class.method``."""
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
        if isinstance(node, ast.ClassDef):
            found.update(
                {f"{node.name}.{m.name}": m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            )
        elif isinstance(node, ast.Assign):
            found.update({t.id: node for t in node.targets if isinstance(t, ast.Name)})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
    return found


def lab_file(checkout: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), "show", f"{COMMIT}:{path}"], capture_output=True, check=True
    ).stdout


def compare(checkout: Path) -> list[str]:
    """Every difference found, as one line each."""
    problems = []
    for ours, (theirs, names) in COPIES.items():
        source = (ROOT / ours).read_text()
        if f"{COMMIT}:{theirs}" not in source:
            problems.append(f"{ours}: does not say it is copied from linker-lab {COMMIT}:{theirs}")
        mine = definitions(source)
        lab = definitions(lab_file(checkout, theirs).decode().replace("linker_lab", "linker_sim"))
        wanted = [n for n in lab if "." not in n] if names == "*" else names.split()
        for name in wanted:
            if name not in lab:
                problems.append(f"{ours}: {name} is not in linker-lab's {theirs}")
            elif name not in mine:
                problems.append(f"{ours}: {name} is missing")
            elif ast.dump(mine[name]) != ast.dump(lab[name]):
                problems.append(f"{ours}: {name} differs from linker-lab's")
    for ours, (theirs, cls, fields) in CONSTANTS.items():
        mine = definitions((ROOT / ours).read_text())[cls]
        lab = definitions(lab_file(checkout, theirs).decode())[cls]
        defaults = lambda node: {  # noqa: E731
            n.target.id: ast.dump(n.value) for n in node.body if isinstance(n, ast.AnnAssign) and n.value
        }
        mine, lab = defaults(mine), defaults(lab)
        for field in fields.split():
            if mine.get(field) != lab.get(field):
                problems.append(f"{ours}: {cls}.{field} differs from linker-lab's")
    for ours, theirs in DATA.items():
        if (ROOT / ours).read_bytes() != lab_file(checkout, theirs):
            problems.append(f"{ours}: differs from linker-lab's {theirs}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.splitlines()[0], file=sys.stderr)
        return 2
    problems = compare(Path(argv[0]).expanduser())
    for line in problems:
        print(line)
    top_level = lambda t: [n for n in definitions(lab_file(Path(argv[0]).expanduser(), t).decode()) if "." not in n]  # noqa: E731
    copied = sum(len(top_level(t)) if n == "*" else len(n.split()) for t, n in COPIES.values())
    print(f"{len(problems)} differences in {copied} copied definitions and {len(DATA)} data files "
          f"(linker-lab {COMMIT})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
