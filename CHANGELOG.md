# Changelog

linker-sim follows [semantic versions](https://semver.org/) from 1.0. The public API is
what [docs/api.md](docs/api.md) lists. A change that breaks it is a new major version, a
removal first warns for one minor release, and a new version of LW-v0's files is a new minor
version.

## 1.0.0 (2026-11-26)

linker-sim is rebuilt: one Linker workstation, LW-v0, on MuJoCo 4.0.1 on the CPU, the same
robot as in linker-lab. The previous multi-robot workspace (0.1.0) is not carried over; it
remains at tag `v0.1.0`. This release is the release candidate below with these changes:

- Licensed under Apache-2.0 (`LICENSE`, `NOTICE`).
- The code copied from linker-lab names linker-lab's release it comes from, `v1.0.0`
  (`tools/check_copies.py`).
- **`ycb-grasp` 0.3.0: a policy trained anywhere on the left half of the table.** **Drop**
  places the object at a random spot in the region the policy was trained on (x 0.2-0.62 m,
  y 0-0.56 m, in `policy.npz` as `demo/region`) instead of one spot; 31 objects (the poses
  grasped in at least 95 % of trials over that region). Copied from linker-lab.
- **The hand's drives no longer lock up.** On MuJoCo 4 a hand drive held at its force limit
  (pressing an object, an impact) lost its implicit damping and oscillated at the limit; the
  motor model read that as a load and stopped the shaft for good, so the finger stayed closed
  (linker-sim-assets issue #28). `HandAction` now keeps each physics step's target so the
  drive never asks for more than its limit less 1 N·m, as `ArmAction` already did (motor
  model `geared-shaft-v1`, as in linker-lab). Grasps now behave as in linker-lab,
  where the policies are trained.
- LW-v0 0.6.0's physics: the thumb yaw's passive damping (4) is on the joint, not in its drive.
- **The first lesson grasps with the gallery's policy** (`examples/pick_up_a_box.py`,
  `docs/lesson.md`): the student's box is handed to `ycb-grasp`'s policy and scene; buttons
  Drop, Grasp, Reset (keys 1-3). The scripted grasp it had (close the hand at a fixed spot)
  only held the box because of the lock-up above; with the hand's real torque it seldom
  holds, as in linker-lab.
- `gallery.load(name)` and `gallery.fetch(name)` are public: an entry's code and files, for
  using its policy from your own code.
- Removed the test of the scripted firm grasp (`test_hand_caps_in_a_firm_grasp`); the
  hand's torque guard has its own test.
- LW-v0's files come from [linker-sim-assets](https://github.com/linker-bot/linker-sim-assets)
  as the package `linker-lw-v0` 1.0.0, installed with linker-sim from the release tag `v1.0.0`
  (`uv.lock` locks its commit), as in linker-lab. Within 1.x linker-sim-assets keeps LW-v0 the
  same robot; `install.sh` checks that `git` works (uv fetches the package with it). Nothing is downloaded at run time any more. The compiled
  robot is the same as bundle 0.2.0's (bodies, joints, drives, couplings, contacts,
  collision shapes, masses, solver options); the visual meshes now have the real colours, the table's work surface green.
- `linker_sim.assets`: `root()` and `path(name)`. Removed: `VERSION`, `BundleError`, the
  bundle download and `LINKER_SIM_BUNDLE`.
- `linker_sim.physics` reads its values from `lw_v0.xml`'s `<option>` (the same values).
- The real controllers' constants are read from the same package (the same values):
  - the O6 motor from `hands/o6/motor.yaml`; `HandActionCfg`'s six constant fields are replaced
    by `motor`, and the passive damping each shaft sees is derived from LW-v0's files
  - the arm limits from `arms/ar5_08/controller.yaml`; `ACCELERATION` and `JERK` are now one
    value per joint, like `VELOCITY`
  - `ARM_JOINTS` and `HAND_JOINTS` come from the two files.

## 1.0.0rc1 (release candidate)

The first release candidate, for testing inside the company. linker-sim simulates LW-v0
on the CPU with MuJoCo 4.0.1, as linker-lab does on the GPU.

**Robot and physics**
- LW-v0 bundle `lw_v0` 0.2.0, the same files as linker-lab's, loaded unchanged:
  - two AR5 arms with O6 hands, the top camera, the support and table
  - 480 Hz physics, the bundle's solver options and home pose
- `Station`: one workstation.
  - adds a floor and your objects
  - steps one decision (1/60 s) at a time with one action per arm and hand
  - `observe()` gives linker-lab's observation; `palm_pose`, `to_base` and the frames
- Arm actions, as the real arm through the xCore SDK:
  - joint or pose commands
  - a jerk-limited trajectory generator (interpolation on) or your own streamed path
    (interpolation off)
  - the real controller's limits and the servo model below the torque caps
- Hand actions: the O6's motors that can't be pushed back, with the presets `open`,
  `close` and `pinch`.
- Objects from meshes (`objects.object_cfg`, `fixed_cfg`, `load_mesh`, `on_table`) with
  linker-lab's mass properties and contact material; `RigidObject.pose`, `velocity` and
  `set_pose`.

**Viewers and tools**
- `viewer.Panel` and `viewer.run`: an interface written once (text, buttons, lists,
  sliders, keys, held keys, markers, a drag handle), shown in MuJoCo Studio (default) or
  viser, at real time. Both serve pages on this computer only.
- `linker-sim doctor`: checks the installation and measures this computer's real-time
  factor.
- `linker-sim play`: the playground. Move a palm goal with the keys (or viser's handle),
  set the hands, read the tool pose, limit interventions and contacts.
- `linker-sim gallery list | info | run`: pretrained policies.
  - `ycb-grasp` 0.2.0 grasps 38 YCB objects with the left hand: 97.9 % of 380 seeded
    grasps here, 97.5–98.3 % in linker-lab.
  - **Its weights are for research use only** (non-commercial; licences in `gallery info`).
- `examples/pick_up_a_box.py`: a first lesson built on the API, explained in
  `docs/lesson.md`.

**Platforms:** macOS on Apple Silicon; Linux on x86_64 and ARM64. No GPU needed.

**Differences from linker-lab**
- MuJoCo 4.0.1 instead of 3.17 (linker-lab's GPU engine requires 3.17). The two agree to
  2e-6 rad, except while a drive is at its torque cap: there MuJoCo 4 adds no implicit
  damping, so joints at a cap move differently.
- Camera images are not simulated.

**Known limitations**
- The minimum requirements are not set yet: the education team's test of this release
  candidate sets them ([requirements](docs/requirements.md)).
- Not tested yet:
  - browsers on integrated GPUs, macOS and Linux ARM64
  - an HTTPS reverse proxy
  - several Studio spectators at once
- Windows and Intel Macs are not supported.
- The bundle and the gallery's files are in a private release: download them by hand
  ([getting started](docs/getting-started.md#1-install)).
