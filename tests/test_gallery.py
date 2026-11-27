"""The gallery and its ``ycb-grasp`` entry, without a browser: the runner (manifests,
artifact download and verification, the CLI) and the gallery's acceptance checks.

The port of ``ycb-grasp`` is checked twice. Against linker-lab: for objects dropped and
settled in linker-lab's scene (Isaac Lab on Newton, GPU; ``tests/data/gallery_reference.npz``,
recorded in linker-lab's scene with linker-lab's copy of this entry), the
same state gives the same observation, action, grasp plan and lift plan here. And in its
own scene: every object comes to rest when dropped, and the scripted grasps of the tomato
soup can, the apple and the mustard bottle end in "grasped".

The entry's artifacts are downloaded into the user cache on first use, from the release the
manifest names.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from linker_sim import __version__, cli, gallery

REFERENCE = Path(__file__).with_name("data") / "gallery_reference.npz"
GRASPED = ("005_tomato_soup_can", "013_apple", "006_mustard_bottle")


# ---------------------------------------------------------------------------- the runner


def test_every_entry_has_a_valid_manifest_for_this_version():
    names = gallery.entries()
    assert "ycb-grasp" in names
    for name in names:
        assert gallery.supported(gallery.manifest(name)), f"{name}: not for {__version__}"


def make_entry(root, name="demo", content=b"weights", **changes):
    (root / name).mkdir(parents=True)
    artifact = root / f"{name}.bin"
    artifact.write_bytes(content)
    sha = hashlib.sha256(b"weights").hexdigest()
    data = dict(name=name, version="0.1.0", title="A demo", description="d", license="l",
                linker_sim=">=0.1.0a1", artifacts={"weights.bin": {"url": artifact.as_uri(), "sha256": sha}})
    data.update(changes)
    data = {k: v for k, v in data.items() if v is not None}
    (root / name / "manifest.yaml").write_text(yaml.safe_dump(data))
    (root / name / "play.py").write_text(
        "RAN = []\ndef run(artifacts, viewer, port):\n    RAN.append((sorted(artifacts), viewer, port))\n"
    )


@pytest.fixture
def fake_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "ROOT", tmp_path / "gallery")
    monkeypatch.setattr(gallery, "cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "gallery"


def test_run_downloads_verifies_and_calls_play_with_the_viewer(fake_gallery, monkeypatch):
    make_entry(fake_gallery)
    loaded = []
    load = gallery.load
    monkeypatch.setattr(gallery, "load", lambda name: loaded.append(load(name)) or loaded[-1])
    gallery.run("demo", viewer="viser", port=8123)
    assert loaded[0].RAN == [(["weights.bin"], "viser", 8123)]
    assert gallery.artifact_path("demo", "weights.bin").read_bytes() == b"weights"


def test_wrong_checksum_is_rejected_and_not_kept(fake_gallery):
    make_entry(fake_gallery, content=b"tampered")
    with pytest.raises(gallery.GalleryError, match="sha256"):
        gallery.fetch("demo")
    assert not gallery.artifact_path("demo", "weights.bin").exists()


def test_unsupported_version_and_bad_manifests(fake_gallery):
    make_entry(fake_gallery, "old", linker_sim="<0.1")
    with pytest.raises(gallery.GalleryError, match="runs on linker-sim <0.1"):
        gallery.run("old")
    make_entry(fake_gallery, "nolicense", license=None)
    with pytest.raises(gallery.GalleryError, match="missing license"):
        gallery.manifest("nolicense")
    with pytest.raises(gallery.GalleryError, match="no gallery entry"):
        gallery.manifest("absent")


def test_cli_list_info_and_run(capsys, monkeypatch):
    assert cli.main(["gallery", "list"]) == 0
    assert "ycb-grasp" in capsys.readouterr().out
    assert cli.main(["gallery", "info", "ycb-grasp"]) == 0
    out = capsys.readouterr().out
    assert "License:" in out and "Runs on linker-sim" in out
    ran = []
    monkeypatch.setattr(gallery, "run", lambda name, viewer, port: ran.append((name, viewer, port)))
    assert cli.main(["gallery", "run", "ycb-grasp", "--viewer", "viser", "--port", "8123"]) == 0
    assert cli.main(["gallery", "run", "ycb-grasp"]) == 0
    assert ran == [("ycb-grasp", "viser", 8123), ("ycb-grasp", "studio", 0)]
    with pytest.raises(SystemExit):
        cli.main(["gallery", "run", "ycb-grasp", "--headless"])  # an entry is for trying, not measuring


# ---------------------------------------------------------------------------- ycb-grasp


@pytest.fixture(scope="module")
def play():
    return gallery.load("ycb-grasp")


@pytest.fixture(scope="module")
def scene(play):
    return play.scene(gallery.fetch("ycb-grasp"))


def index(scene, name: str) -> int:
    return next(i for i, o in enumerate(scene.items) if o["name"] == name)


def run_until(scene, states, limit):
    for _ in range(limit):
        scene.step()
        if scene.state in states:
            return


def test_station_holds_every_object_parked_below_the_floor(play, scene):
    """Every object (31), one free body each, parked under the floor where nothing touches them."""
    from linker_sim import workstation

    assert len(scene.items) == 31 and len(scene.station.objects) == 31
    scene._reset()
    for i, o in enumerate(scene.objects):
        pos, _ = o.pose()
        assert np.allclose(pos, play.parking(i)) and pos[2] + 0.3 < workstation.FLOOR_TOP - 0.05
    for _ in range(2 * len(scene.items)):  # two parking rounds: none falls far, none touches
        scene.step()
        assert all(o.pose()[0][2] > -2.0 - 2.5 for o in scene.objects)
    bodies = {scene.station.model.body(f"item_{i:02d}").id for i in range(len(scene.items))}
    model, data = scene.station.model, scene.station.data
    assert not any({model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & bodies for c in data.contact[: data.ncon])


def test_observation_action_and_plans_equal_linker_labs(scene):
    """The state linker-lab's scene saw at Grasp gives the observation, action, grasp plans
    (the policy's and three random actions') and lift plans it recorded.

    Tolerances: the observation is float32, and linker-lab reads the joint ranges as
    float32 (here: the float64 values of LW-v0's files), about 1e-7 apart. linker-lab's actor ran
    its convolutions in TF32 (cuDNN's default on the GPU), so its action is up to 2e-4 from
    float32 (at most 16 µm of palm translation); the same network on the GPU in float32
    agrees with the CPU to 5e-7. The plans are float64 IK from the same inputs: this code on
    the GPU reproduces the recorded plans to 2e-14 rad. On the CPU the policy's plans agree to
    1e-6 rad (0.1 µm at the palm); a random action's plan can differ more (up to 7e-6 rad
    measured with ycb-grasp 0.3.0, at ticks where the IK ends its 50 iterations before it has
    converged), where the processor's rounding changes where it stops. The hand's blend into the
    policy's shaft offsets is float32 (the copied ``plan`` divides integer ticks), and its
    polynomial cancels to 1 at the end of closing, so processors differ by up to 1e-6 rad
    there. Plans the arm can't follow
    (invalid) may amplify rounding; they are never executed, so only their validity is
    compared."""
    ref = np.load(REFERENCE)
    for n, name in enumerate(ref["names"]):
        scene._reset()
        scene._drop(index(scene, str(name)))  # the object's surface points and bounds
        f64 = dict(dtype=torch.float64)
        scene.settled_pos = torch.tensor(ref["pos"][n], **f64)[None]
        scene.settled_quat = torch.tensor(ref["quat"][n], **f64)[None]
        state = (torch.tensor(ref[k][n], **f64)[None] for k in ("q", "qd", "shaft", "vel"))
        obs = scene.observation(*state)
        assert np.abs(obs.numpy()[0] - ref["obs"][n]).max() < 1e-6, name
        action = scene.actor(torch.tensor(ref["obs"][n])[None])
        assert np.abs(action.numpy()[0] - ref["action"][n]).max() < 1e-3, name
        for j, a in enumerate(ref["actions"][n]):
            arm, hand, valid = scene.plan(torch.tensor(a)[None])
            assert bool(valid[0]) == bool(ref["valid"][n, j]), (name, j)
            if valid[0]:
                tolerance = 2e-6 if j == 0 else 1e-5  # the policy's action, then random ones
                assert np.abs(arm.numpy()[0] - ref["arm"][n, j]).max() < tolerance, (name, j)
            assert np.abs(hand.numpy()[0] - ref["hand"][n, j]).max() < 5e-6, (name, j)
            lift, lift_valid = scene.plan_lift(torch.tensor(ref["arm"][n, j, -1:]))
            assert bool(lift_valid[0]) == bool(ref["lift_valid"][n, j])
            assert np.abs(lift.numpy()[0] - ref["lift"][n, j]).max() < 2e-6, (name, j)


def test_every_object_comes_to_rest_when_dropped(scene):
    """Each object, dropped at a random spot in the region in a random resting pose and yaw, is at rest on the
    table after the settling ticks, resting on it: the table carries its weight (the contact
    force on it from anything but the hand, as the grasp's check reads it). A round object
    still rocking slowly within the rest thresholds carries part of it in its motion (the
    pear: 6%)."""
    scene.rng = np.random.default_rng(0)
    not_at_rest, weights = [], []
    for i, o in enumerate(scene.items):
        scene.press("drop", i)
        run_until(scene, {"ready", "empty"}, 400)
        if scene.state != "ready":
            not_at_rest.append(o["name"])
            continue
        weights.append(scene.other_contact_force()[2] / (o["mass"] * 9.81))
    assert not_at_rest == []
    assert abs(np.median(weights) - 1) < 1e-6 and np.allclose(weights, 1.0, atol=0.1), weights


@pytest.mark.parametrize("name", GRASPED)
def test_scripted_grasp_ends_grasped(scene, name):
    """Drop, Grasp: the episode runs to "grasped" (lifted 10 cm, held 2 s, touching nothing
    but the hand); then Reset clears the table. Seed 5 drops each mid-region; near the
    region's far corner (seed 1: x 0.26, y 0.53, at the arm's reach) a grasp can go either way
    with the timing of the Grasp press."""
    scene.rng = np.random.default_rng(5)
    scene.press("drop", index(scene, name))
    run_until(scene, {"ready", "empty"}, 400)
    assert scene.state == "ready", scene.message
    scene.press("grasp")
    run_until(scene, {"done"}, 800)
    assert scene.message.endswith(": grasped."), scene.message
    assert np.linalg.norm(scene.other_contact_force()) < 0.01  # in the hand only
    scene.press("reset")
    scene.step()
    assert scene.state == "empty" and scene.active is None


def test_cli_reports_errors_without_a_traceback(capsys):
    """A missing entry (or a file that can't be downloaded) is one line on stderr, exit 1."""
    from linker_sim import cli

    assert cli.main(["gallery", "info", "no-such-entry"]) == 1
    assert capsys.readouterr().err.startswith("linker-sim: no gallery entry 'no-such-entry'")
