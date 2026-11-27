# A first lesson: pick up a box

This page builds a small lesson on linker-sim's API, one piece at a time. A student gives a
box of their own to the gallery's grasp policy and runs it with three buttons: drop the box
on the table, grasp it, reset. On the way it shows the parts of the API a lesson needs most:

- an object, described the way a policy sees it
- a gallery entry's policy and scene, used from your own code
- reading the palm point's and an object's pose
- a panel that runs in either viewer

The finished program is
[`examples/pick_up_a_box.py`](../examples/pick_up_a_box.py):

```bash
uv run python examples/pick_up_a_box.py                  # MuJoCo Studio: open the address it prints
uv run python examples/pick_up_a_box.py --viewer viser   # the same lesson in viser
uv run python examples/pick_up_a_box.py --check          # no viewer: drop, grasp, check the lift
```

You can also paste the pieces below into `uv run python`, in order. Every piece runs as it
is, and a test runs this page.

Why a policy and not a script that moves the hand? The O6's motors close slowly and only as
hard as they really can, so a hand that simply closes around an object at a fixed spot
seldom holds it: where the palm goes and how each finger closes has to fit the object. The
gallery's policy was trained for exactly that, on thousands of objects.

## 1. The box

An object is its collision shape (any mesh: `objects.load_mesh("mug.stl")` reads one from a
file), its mass, and, for the policy, 512 points on its surface, its bounds and how it rests
on the table. This box is 6 × 6 × 10 cm and 350 g, and rests upright.

```python
import numpy as np
import trimesh

from linker_sim import gallery, objects, physics, viewer
from linker_sim.viewer import Panel, Sphere

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
```

## 2. The policy and its scene

`gallery.load` gives an entry's code and `gallery.fetch` its files (downloaded once and
checked). The `ycb-grasp` entry has the policy, a station that holds the objects, and the
episode the policy was trained with (`GraspScene`): the robot at home, the object dropped
where the policy looks for it with a random turn, one look, then the grasp it planned, a
15 cm lift and a 2 s hold. Use it as it is: change the episode and the policy no longer sees
or does what it learned.

```python
entry = gallery.load("ycb-grasp")
files = gallery.fetch("ycb-grasp")  # downloaded once, checked against their sha256
actor, demonstration = entry.load_policy(files["policy.npz"])
station = entry.station([box_item])
scene = entry.GraspScene(station, [box_item], actor, demonstration)
box = station.objects["item_00"]
table_top = objects.table()["top"]
```

## 3. One decision, and what to show

`step()` runs one **decision** (1/60 s) of the scene. `palm_pose` gives the palm point's
position and orientation in the station frame (x forward over the table, z up; quaternions
x, y, z, w), and `pose()` an object's.

```python
def step():
    """One decision of the scene: settling the box, or the grasp the policy planned."""
    scene.step()


def box_height():
    """How high the box's bottom is above the table [m] (0 while it is parked below the floor)."""
    return max(0.0, box.pose()[0][2] - 0.05 - table_top) if scene.active is not None else 0.0


def palm_to_box():
    """The distance from the palm point to the box's centre [m]."""
    return float(np.linalg.norm(station.palm_pose("left")[0] - box.pose()[0]))
```

Try it without a viewer. Buttons are pressed with `scene.press` and take effect at the next
decision; 60 decisions are one simulated second, and the computer runs them faster than
real time:

```python
for button, seconds in [("drop", 3), ("grasp", 12)]:
    scene.press(button, 0)  # the box is the scene's object 0
    for _ in range(seconds * physics.DECISION_HZ):
        step()
print(scene.message, f"Box above the table: {100 * box_height():.1f} cm")  # Box: grasped. ... about 16 cm
scene.press("reset", 0)
step()
```

## 4. The panel

A `Panel` describes the interface once; both viewers show it. Text lines are functions,
called again after every decision. Buttons call a function, and `key=` puts them on a key
too. Markers are drawn in the scene.

```python
panel = Panel("Lesson: pick up a box")
panel.text(lambda: scene.message)
panel.text(lambda: f"Palm to box: {1000 * palm_to_box():.0f} mm" if scene.active is not None else "Palm to box: -")
panel.text(lambda: f"Box above the table: {100 * box_height():.1f} cm")
panel.button("Drop", lambda: scene.press("drop", 0), key="1")  # the viewers show the key
panel.button("Grasp", lambda: scene.press("grasp"), key="2")
panel.button("Reset", lambda: scene.press("reset"), key="3")
panel.markers(lambda: [Sphere(station.palm_pose("left")[0], 0.01)])  # the palm point
```

## 5. Run it

`viewer.run` steps the station at real time, calling `step()` once per decision, and
serves the page on this computer. Open the address it prints in a browser; Ctrl+C stops
it.

```python
viewer.run(station, panel, step=step)  # or viewer="viser"
```

Press 1 and wait until the box has come to rest, then 2: the arm reaches for the box, the
hand closes and lifts it, and the panel shows the palm closing in and the box rising. Press
3 to start again.

Every callback, the buttons and text lines and `step` itself, runs on the same thread as
the simulation, between two decisions. So a callback may read and command the station
directly; there is nothing to lock.

## What to try next

- Change the box: a slider for its width (build the scene again when it changes), or a
  mesh of your own with `objects.load_mesh`. Which ones does the policy hold?
- Show whether the hand touches the box. `station.data.contact` lists MuJoCo's contacts;
  reading it is allowed, writing to MuJoCo is not supported.
- Move the hand yourself: a station whose left arm takes poses of its palm point,
  `Station(arm_left=arm_action_cfg("left", command="pose", frame="palm"))`, moves the hand
  to a goal you set (`station.to_base` turns a goal in the station frame into the arm's
  base frame, the frame the real controller uses).

The [API reference](api.md) lists everything you can use, and the
[viewers guide](viewers.md) explains what differs between Studio and viser.
