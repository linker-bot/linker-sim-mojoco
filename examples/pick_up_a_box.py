"""Lesson: pick up a box with the left hand, grasped by a trained policy.

A small lesson built on linker-sim's public API; ``docs/lesson.md`` builds it step by step.
The student gives a box of their own to the gallery's grasp policy (``ycb-grasp``) and runs
it with the panel's buttons (or keys 1-3): drop the box on the table, grasp it, reset. The
panel shows what the scene is doing, how far the palm point is from the box and how high the
box is.

    uv run python examples/pick_up_a_box.py                  # MuJoCo Studio
    uv run python examples/pick_up_a_box.py --viewer viser   # the same lesson in viser
    uv run python examples/pick_up_a_box.py --check          # no viewer: drop, grasp, check the lift
"""

import argparse

import numpy as np
import trimesh

from linker_sim import gallery, objects, physics, viewer
from linker_sim.viewer import Panel, Sphere

# 1. The box, as the policy sees an object.
mesh = trimesh.creation.box((0.06, 0.06, 0.10))  # 6 x 6 cm, 10 cm high, centred on its origin
parts = [(mesh.vertices, mesh.faces)]
com, inertia = objects.mass_properties(parts, mass=0.35)
points, _ = trimesh.sample.sample_surface_even(mesh, 512, seed=0)  # what the policy looks at
box_item = {
    "name": "lesson_box",
    "label": "Box",
    "parts": parts,
    "mass": 0.35,
    "com": com,
    "inertia": inertia,
    "points": np.asarray(points, dtype=np.float32),
    "bounds": mesh.bounds,
    "poses": [((0.0, 0.0, 0.0, 1.0), 0.05)],  # resting upright: its origin 5 cm above the table
}

# 2. The policy and its scene: a station with the box, the episode it was trained with.
entry = gallery.load("ycb-grasp")
files = gallery.fetch("ycb-grasp")  # downloaded once, checked against their sha256
actor, demonstration = entry.load_policy(files["policy.npz"])
station = entry.station([box_item])
scene = entry.GraspScene(station, [box_item], actor, demonstration)
box = station.objects["item_00"]
table_top = objects.table()["top"]


# 3. One decision, and what to show.
def step():
    """One decision of the scene: settling the box, or the grasp the policy planned."""
    scene.step()


def box_height():
    """How high the box's bottom is above the table [m] (0 while it is parked below the floor)."""
    return max(0.0, box.pose()[0][2] - 0.05 - table_top) if scene.active is not None else 0.0


def palm_to_box():
    """The distance from the palm point to the box's centre [m]."""
    return float(np.linalg.norm(station.palm_pose("left")[0] - box.pose()[0]))


# 4. The panel.
panel = Panel("Lesson: pick up a box")
panel.text(lambda: scene.message)
panel.text(lambda: f"Palm to box: {1000 * palm_to_box():.0f} mm" if scene.active is not None else "Palm to box: -")
panel.text(lambda: f"Box above the table: {100 * box_height():.1f} cm")
panel.button("Drop", lambda: scene.press("drop", 0), key="1")  # the viewers show the key
panel.button("Grasp", lambda: scene.press("grasp"), key="2")
panel.button("Reset", lambda: scene.press("reset"), key="3")
panel.markers(lambda: [Sphere(station.palm_pose("left")[0], 0.01)])  # the palm point


def check() -> bool:
    """Drop, wait until the box rests, grasp, wait for the result; True if it was grasped."""
    scene.rng = np.random.default_rng(0)
    scene.press("drop", 0)
    for _ in range(10 * physics.DECISION_HZ):
        step()
        if scene.state == "ready":
            break
    scene.press("grasp")
    for _ in range(20 * physics.DECISION_HZ):
        step()
        if scene.state == "done":
            break
    return scene.message.endswith(": grasped.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Lesson: pick up a box with the left hand.")
    parser.add_argument("--viewer", choices=viewer.VIEWERS, default="studio")
    parser.add_argument("--check", action="store_true", help="no viewer: drop, grasp, check the lift")
    args = parser.parse_args(argv)
    if args.check:
        grasped = check()
        print(f"{scene.message} Box above the table: {100 * box_height():.1f} cm. {'passed' if grasped else 'FAILED'}")
        return 0 if grasped else 1
    viewer.run(station, panel, step=step, viewer=args.viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
