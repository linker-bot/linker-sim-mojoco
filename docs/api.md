# API reference

This page is linker-sim's public API: the modules, names and members listed here. From
1.0 they follow semantic versions:

- a change that breaks code using them is a new major version;
- a name to be removed first warns for one minor release;
- a new version of LW-v0's files (the pinned `linker-lw-v0`) is a new minor version, its
  changes listed in the [changelog](../CHANGELOG.md).

Everything else is internal and may change in any release, whether or not its name starts
with an underscore. That includes the modules `engine`, `playground` and `cli` and the
viewer front ends. Each public module's `__all__` is exactly its list below, and a test
keeps the two equal.

**Conventions.**
- Units: metres, radians, seconds, kilograms, newton metres.
- Quaternions: x, y, z, w, as in linker-lab. MuJoCo's own w, x, y, z order appears only
  in `station.model` and `station.data`.
- Poses are in the **station frame** (x forward over the table, z up, origin between the
  two arm mounts, table top 0.5 m below it). Arm pose commands are the exception: they are
  in **each arm's base frame**, the frame the real controller uses (`Station.to_base`).
- A **decision** is 1/60 s: one command per action, held for 8 physics steps of 1/480 s.
- **Reading MuJoCo directly is allowed; writing to it is not supported.**
  `station.model` and `station.data` are MuJoCo's `MjModel` and `MjData`. Read them for
  anything this API lacks. Writing to them bypasses the drive models, and what it does
  may change in any release.

```python
import linker_sim

print(linker_sim.__version__)
```

## `linker_sim.workstation`

The station, its frames and its action configurations.

### `Station(objects=None, floor=True, *, arm_left=None, arm_right=None, hand_left=None, hand_right=None)`

One LW-v0 workstation simulated on the CPU. It loads LW-v0's scene `lw_v0.xml`
unchanged: the two arm units, the top camera, the support with its table, the solver
options and the home keyframe. It adds the floor (`floor=True`) and your objects
(`objects`: name → `objects.object_cfg(...)` or `objects.fixed_cfg(...)`; `"floor"` is
taken). It makes one action per arm and hand from the configurations given:
`arm_action_cfg(side, ...)` and `hand_action_cfg(side, ...)`. By default the arms take
joint goals and the hands take shaft targets in radians. A new station is at home.

- `step(*, arm_left=None, arm_right=None, hand_left=None, hand_right=None)`: one
  decision. Each action takes its command, then the physics advances 8 steps.
  - An arm's command is 7 joint positions [rad] by default (14 with
    `stream="points_velocity"`, 112 with `stream="samples"`). With `command="pose"` it is
    a position and a quaternion, 7 values, in the arm's base frame.
  - A hand's command is 6 shaft targets in `hand.SHAFTS` order.
  - An omitted command repeats the previous one (after a reset: hold home). A non-finite
    value (`nan`) keeps that value of the previous command.
  - A stream (`interpolation=False`) is your own path: send its next point every decision.
- `reset()`: back to the home keyframe. The robot is at home with every drive holding it,
  the objects are back at their initial poses and at rest, and every action holds home.
- `observe()`: an array (52,): the positions [rad] of the 13 driven joints of the left
  unit, then of the right unit, then their velocities [rad/s] in the same order. Per unit:
  `ARM_JOINTS[side]`, then `HAND_JOINTS[side]`. This is linker-lab's observation.
- `palm_pose(side)`: position (3,) and quaternion (4,) of the palm point of `"left"` or
  `"right"`, station frame. The orientation is the hand base frame's (see `PALM_OFFSET`).
- `to_base(side, pos, quat)`: a station-frame pose expressed in that arm's base frame,
  the frame its pose commands are given in. E.g. the palm 5 cm higher:
  `station.to_base("left", pos + (0, 0, 0.05), quat)`.
- `objects`: name → `RigidObject` for every free object you added.
- `actions`: `"arm_left"`, `"arm_right"` (`arm.ArmAction`), `"hand_left"`, `"hand_right"`
  (`hand.HandAction`), in the order they are applied, for reading their state.
- `time`: simulated seconds since the last reset.
- `model`, `data`: MuJoCo's `MjModel` and `MjData`, for reading. In MuJoCo every name of
  an arm unit is prefixed with its side, e.g. `left/hand_lh_hand_base_link`.

### `RigidObject`

A free object of a station (`station.objects[name]`).

- `name`: its name, also its MuJoCo body's name.
- `pose()`: position (3,) and quaternion (4,) of the object frame, station frame.
- `velocity()`: linear velocity (3,) of the centre of mass [m/s] and angular velocity (3,)
  [rad/s], station frame.
- `set_pose(pos, quat=(0, 0, 0, 1))`: put the object at a pose, at rest. `pose()` reads
  the new pose at once, and the next `step()` starts from it. The solver's starting guess
  for the object is cleared too, so what follows depends only on where it was put.

### `arm_action_cfg(side, *, frame="tcp", **options)`

An `arm.ArmActionCfg` for one arm. `options` are its fields: `command`, `interpolation`,
`stream`, `limits`, `ik_damping`. With `command="pose"`, `frame` is the frame that is
commanded: `"tcp"` is the arm's tool point (the frame the real controller reports), and
`"palm"` is the hand's palm point (`PALM_OFFSET`).

### `hand_action_cfg(side, **options)`

A `hand.HandActionCfg` for one hand; `options` are its fields, e.g. `normalized=True`.

### `SIDES`

`("left", "right")`.

### `ARM_JOINTS`

Side → the names of the seven arm joints J1..J7, without the side prefix.

### `HAND_JOINTS`

Side → the names of the six driven hand joints, in `hand.SHAFTS` order. The other five
hand joints (thumb ip, the four dip joints) follow them through fixed couplings.

### `TCP_BODY`

Side → the body of the arm's tool point (the flange face).

### `PALM_BODY`

Side → the hand's base link, the body the palm point is on.

### `PALM_OFFSET`

`(0.055, 0.0, 0.055)` [m]: the palm point in the hand base link's frame, the same on both
hands. That frame's x axis is the palm normal (the fingers flex towards +x), and z points
from the wrist to the knuckles. The point is 3.5 cm in front of the palm, halfway between
wrist and knuckles.

### `HOME`

Side → joint name → home angle [rad], for all 18 joints of each unit: palms down about
0.3 m above the table, hands open, thumbs turned in.

### `FLOOR_TOP`

`-1.267` [m]: the height of the floor's top in the station frame.

## `linker_sim.arm`

How the AR5 arm is commanded, as on the real arm through the xCore SDK.

- **Interpolation on** (default; the SDK's PP tracking): each command is a goal. A
  jerk-limited trajectory generator moves toward the latest goal, all seven joints
  arriving together, within the real limits.
- **Interpolation off** (the SDK's CSP servo): each command is the next point of your own
  path, joined by a cubic curve.

A servo model keeps every drive 1 N·m below its torque cap (120 N·m for J1–J4, 100 N·m for
J5–J7). Pose commands are turned into joint positions by one damped-least-squares step
per decision, as in linker-lab.

### `ArmActionCfg`

The options of one arm's action (a dataclass; make it with
`workstation.arm_action_cfg`).

- `command`: `"joint"` (default): 7 joint positions [rad]. `"pose"`: a position [m] and a
  quaternion of the commanded frame, in the arm's base frame.
- `interpolation`: `True` (default): the command is a goal for the trajectory generator.
  `False`: the command is the next point of your path.
- `stream`: with `interpolation=False`:
  - `"points"` (default): 7 positions. The point's velocity is the difference to the
    previous point, so the path never stops at a point.
  - `"points_velocity"`: 7 positions and 7 velocities.
  - `"samples"`: the 8 per-step positions, then the 8 per-step velocities (112 values),
    used as given.
- `limits`: `"real"` (default): the real controller's limits (`VELOCITY`, `ACCELERATION`,
  `JERK`, the joint ranges, `WRIST_OCTAGON`). `None`: no limits, only with
  `interpolation=False`.
- `ik_damping`: the damping of the pose commands' least-squares step (default 0.01).
- `asset_name`, `joint_names`, `body_name`, `body_offset`: the arm unit, its joints and
  the commanded frame, filled in by `arm_action_cfg`.

### `ArmAction`

One arm's action (`station.actions["arm_left"]`). The station calls it; read its state:

- `limit_hits`: a tensor (1,): decisions and physics steps since the last reset in which a
  limit changed the command, or the jerk limit was exceeded. A policy that relies on
  motion the real arm can't make shows up here.
- `reference`: the reference (positions, velocities) of the current physics step, (1, 7)
  each.
- `raw_actions`: the last command (1, `action_dim`), non-finite values replaced.
- `processed_actions`: the same.
- `action_dim`: the number of values a command has.
- `cfg`: its `ArmActionCfg`.
- `process_actions(actions)`, `apply_actions()`, `reset()`: called by the station (once
  per decision, before each physics step, after its reset), as Isaac Lab's action manager
  calls linker-lab's actions. Don't call them yourself.

### `VELOCITY`

The maximum joint velocities [rad/s] J1..J7: `(0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49)`,
the xCore SDK's motion profile of the AR5_08. This and the next three are read from
linker-sim-assets' `arms/ar5_08/controller.yaml`.

### `ACCELERATION`

The maximum joint accelerations [rad/s²] J1..J7: 10 each.

### `JERK`

The maximum joint jerks [rad/s³] J1..J7: 40 each.

### `WRIST_OCTAGON`

The region the real controller allows for (J6, J7) [rad]: the vertices of a convex
octagon, in order.

### `TORQUE_GUARD_MARGIN`

1.0 N·m: how far below each joint's torque cap the servo model keeps the drive.

## `linker_sim.hand`

The O6 hand. Each of its six driven joints has a motor behind a gear that can't be pushed
back: the shaft moves toward its target, slowly and only as far as its torque allows, and
a load can stop it but never push it back. So a grasp holds without any command. Each
physics step the drive's target is kept so the drive never asks for more than its force limit
less 1 N·m (`arm.TORQUE_GUARD_MARGIN`), as for the arm (motor model
`geared-shaft-v1`).

### `HandActionCfg`

The options of one hand's action (a dataclass; make it with
`workstation.hand_action_cfg`).

- `normalized`: `False` (default): targets in radians. `True`: in [-1, 1] over each
  joint's range.
- `motor`: the motor model's constants, read from linker-sim-assets' `hands/o6/motor.yaml`
  (its shafts, `motor_stiffness`, `advancing_cap`, `output_damping`, `viscosity`,
  `speed_limit`, and the `passive_damping` each shaft sees, derived from LW-v0's files).
  They model the real O6; change them only to study the model.
- `asset_name`, `joint_names`, `motor`: filled in by `hand_action_cfg`.

### `HandAction`

One hand's action (`station.actions["hand_left"]`). The station calls it; read its state:

- `shaft`: the motor shaft positions [rad], (1, 6).
- `processed_actions`: the shaft targets in effect [rad], (1, 6). Targets outside a
  joint's range are clamped into it.
- `raw_actions`: the last command (1, 6).
- `action_dim`: 6.
- `cfg`: its `HandActionCfg`.
- `process_actions(actions)`, `apply_actions()`, `reset()`: called by the station.

### `SHAFTS`

The six driven joints in command order: thumb pitch, thumb yaw, index, middle, ring,
pinky.

### `PRESETS`

Name → six shaft targets [rad]: `"open"`, `"close"` (the thumb turned in and every finger
closed) and `"pinch"` (thumb and index).

### `THUMB_OPPOSE_YAW`

1.04 rad: the thumb yaw that turns the thumb opposite the index finger.

### `thumb_oppose(current)`

The shaft targets `current` (a tensor (N, 6)) with the thumb yaw set to
`THUMB_OPPOSE_YAW`.

## `linker_sim.objects`

Rigid objects made of mesh parts. Each part collides as its convex hull, with the object
contact material `MATERIAL`.

### `object_cfg(parts, mass, *, com=None, inertia=None, pos=(0.35, 0, 0), rot=(0, 0, 0, 1), visual=None)`

A free object at `pos` / `rot` (station frame). Without `com` and `inertia` they are
computed for a uniform density over the parts' hulls. `visual` is a mesh `Part` shown
instead of the parts; it never collides.

### `fixed_cfg(parts, *, pos=(0, 0, 0), rot=(0, 0, 0, 1))`

A fixed object: it collides with the same material but never moves (a shelf, a bin).

### `load_mesh(path, scale=1.0)`

A mesh file (STL, OBJ, PLY, ...) as a list with one `Part`, in metres after `scale`.

### `mass_properties(parts, mass)`

The centre of mass (3,) and the inertia (3, 3) about it [kg m²], for a uniform density
over the parts' hulls.

### `on_table(parts, xy, rot=(0, 0, 0, 1), clearance=0.001)`

The position that puts the object, turned by `rot`, `clearance` above the table top at
`xy`. Released there, it drops onto the table.

### `table()`

The table's collision box: `{"top": height, "center": (x, y), "half_size": (x, y)}` [m].

### `ObjectCfg`

What `object_cfg` and `fixed_cfg` return: `parts`, `mass`, `com`, `inertia` (None for a
fixed object), `pos`, `rot`, `visual`.

### `Part`

One mesh part: a tuple of vertices (N, 3) [m] in the object frame and triangle indices
(M, 3).

### `MATERIAL`

The contact settings of every object surface: friction 1, torsional friction 0.005,
condim 4, and MuJoCo's solref and solimp. Where an object touches the hand, the hand's
friction (4) applies.

## `linker_sim.physics`

The timing and solver LW-v0 is simulated with, the same values as linker-lab's. They are
read from the `<option>` of LW-v0's `lw_v0.xml`, which MuJoCo also loads.

### `PHYSICS_HZ`

480 physics steps per second; collision detection runs in every one.

### `DECISION_HZ`

60 decisions per second.

### `DECIMATION`

8 physics steps per decision.

### `SOLVER`

The solver settings, with the names of MJCF's `<option>`.

## `linker_sim.viewer`

Show a station and an interface in a viewer. The interface is a `Panel`, written once; it
runs in MuJoCo Studio (the default) or in viser. The [viewers guide](viewers.md) explains
how they differ.

### `run(station, panel=None, *, step=None, viewer="studio", port=0, camera=None, seconds=None)`

Show `station` and `panel` in `viewer` (`"studio"` or `"viser"`) and step the station at
real time. It runs until you stop it (Ctrl+C) or until `seconds` of simulated time have
passed, and returns `Stats`.

- `step()`: one decision; by default `station.step()`. Give your own to command the
  station, e.g. `step=lambda: station.step(hand_left=command["hand"])`.
- `port`: 0 takes the first free port from 8080.
- `camera`: the initial view, `(eye, look_at)` [m] in the station frame.

Every callback of the panel, and `step`, runs on the thread that runs `run`, between two
decisions. So callbacks may read and command the station directly, without locks.

The page is served on this computer only (`127.0.0.1`); it prints its address.

### `Panel(title="linker-sim")`

An interface: a titled panel of elements, shown in the order they are added, plus keys,
markers and a drag handle. Build it before `run` shows it: the elements are fixed while it
runs, while the text of a line, the markers and the handle's pose are read anew after
every decision.

- `text(line)`: a line of text; `line()` returns it.
- `button(label, on_click, *, key=None)`: a button, optionally also on a key.
- `select(label, options, on_select, *, index=0)`: a drop-down list; choosing calls
  `on_select(i)`.
- `slider(label, low, high, on_change, *, value=None)`: a slider; moving it calls
  `on_change(value)`.
- `key(key, on_press)`: call `on_press()` when `key` is pressed.
- `hold(*keys)`: report whether `keys` are held down, in `held`; include `"shift"` to read
  Shift with them. In viser a press counts as the key held for 0.1 s, extended by the
  keyboard's auto-repeat (see the [viewers guide](viewers.md)).
- `held`: the keys of `hold` that are held now.
- `view`: the direction the camera looks in (a unit vector, station frame), e.g. for keys
  that move relative to the view.
- `markers(draw)`: `draw()` returns the markers (`Sphere`, `Axes`) to draw in the scene.
- `handle(pose, on_move)`: a drag handle for a pose, **viser only**: drawn at `pose()`
  (position, quaternion); dragging it calls `on_move(pos, quat)`. Offer the same through
  keys or sliders for Studio.
- `title`: its title.

### `Sphere(pos, radius=0.01, rgba=(0.1, 0.8, 0.2, 0.6))`

A sphere marker; colour channels 0..1.

### `Axes(pos, quat=(0, 0, 0, 1), length=0.1)`

The axes of a frame at a pose: x red, y green, z blue.

### `KEYS`

The keys a panel can use: the letters, the digits, `"tab"`, `"space"` and `"shift"` (with
`hold` only). viser moves its camera with W, A, S, D, Q, E and the arrow keys, so a panel
for both viewers leaves those letters free.

### `Stats`

What `run` returns: `decisions` (stepped), `speed` (simulated seconds per second) and
`capacity` (how much faster than real time the loop could run). Every panel's last line
shows the speed and the capacity.

### `VIEWERS`

`("studio", "viser")`.

## `linker_sim.gallery`

Pretrained policies that run on LW-v0 (`linker-sim gallery`).

**An entry** is a folder in the repository's `policy-gallery/` with two files:
- `manifest.yaml`: `name` (the folder's name), `version`, `title`, `description`,
  `license`, `linker_sim` (the linker-sim versions it runs on, a version specifier such as
  `">=1.0,<2"`) and `artifacts` (file name → `url` and `sha256`).
- `play.py`: `run(artifacts, viewer="studio", port=0)`, which opens the entry's scene with
  `viewer.run`; `artifacts` maps each file name to its path.

The files are cached in `~/.cache/linker-lab/gallery/<entry>/`, shared with linker-lab. A
file put there by hand is used if its sha256 matches.

### `entries()`

The names of the gallery's entries.

### `manifest(name)`

An entry's manifest as a dict, checked.

### `info(name)`

A readable summary of an entry: its description, licence, the linker-sim versions it runs
on, and its files.

### `supported(manifest)`

Whether this linker-sim's version is in the manifest's range.

### `run(name, viewer="studio", port=0)`

Open an entry in a viewer. It downloads its files on first use, checks them against their
sha256, and returns when stopped.

### `load(name)`

An entry's `play.py` as a module, for using its policy and scene from your own code (as
[the lesson](lesson.md) does with `ycb-grasp`). What a module holds is the entry's own; see
its docstring.

### `fetch(name)`

An entry's files: a dict from each file name to its path in the cache, downloaded on first
use and checked against their sha256.

### `GalleryError`

An entry is missing, malformed, not for this version, or its files don't match.

## `linker_sim.assets`

LW-v0's robot files: the folder `assemblies/lw_v0/` (and the parts it is built from) of
[linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) (MJCF and URDF with
their meshes), installed with linker-sim as the data package `linker-lw-v0` from the release
tag `v1.0.0` (named in `pyproject.toml`; `uv.lock` locks its commit); the same files linker-lab
uses. Within 1.x linker-sim-assets keeps LW-v0 the same robot (paths, names, physics, the
controllers' constants, the visual meshes' shape and place). Nothing is downloaded at run time.

### `NAME`

`"lw_v0"`.

### `root()`

The installed `assemblies/lw_v0/` folder.

### `path(name)`

The path of one of LW-v0's files, e.g. `path("lw_v0.xml")` (the scene a station loads) or
`path("left.xml")`; `FileNotFoundError` if there is no such file.

## Command line

| Command | What it does |
| --- | --- |
| `linker-sim doctor` | Checks the installation: versions, Warp and torch on the CPU, one simulated second (its real-time factor) |
| `linker-sim play [--viewer studio\|viser] [--port N]` | The playground: drive LW-v0 by hand |
| `linker-sim gallery list` | The gallery's entries |
| `linker-sim gallery info ENTRY` | An entry's description and licences |
| `linker-sim gallery run ENTRY [--viewer studio\|viser] [--port N]` | Open an entry |
