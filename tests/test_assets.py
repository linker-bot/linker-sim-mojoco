"""LW-v0's files from the package.

Copied from linker-lab v1.0.0:tests/test_assets.py (the tests that do not convert to USD;
linker-sim does not use USD); change them there first.
"""

from pathlib import Path

import pytest

from linker_sim import assets


def test_files_come_from_the_package():
    assert assets.root().name == "lw_v0" and assets.path("lw_v0.xml").is_file()
    assert (assets.root() / "lw_v0.urdf").is_file()
    with pytest.raises(FileNotFoundError):
        assets.path("layout.yaml")


def test_the_installed_files_are_the_pinned_release():
    """pyproject.toml installs LW-v0's files from a 1.x release tag of linker-sim-assets, and the
    installed package is that release (a stale environment fails here). Within 1.x LW-v0 stays
    the same robot (linker-sim-assets' promise), so moving to another 1.x release changes only
    pyproject.toml and uv.lock; 2.0.0 would be a product decision."""
    import importlib.metadata
    import re
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    (pin,) = [d for d in project["dependencies"] if d.startswith("linker-lw-v0 ")]
    tag = re.fullmatch(r"linker-lw-v0 @ git\+https://github\.com/linker-bot/linker-sim-assets@v(1\.\d+\.\d+)", pin)
    assert tag, f"pyproject.toml pins {pin!r}, not a 1.x release tag"
    assert importlib.metadata.version("linker-lw-v0") == tag[1]


# ---------------------------------------------------------------- controller data


@pytest.mark.parametrize("side", ["left", "right"])
def test_motor_constants_and_derived_passive_damping(side):
    """motor.yaml read for one hand; the passive damping derived from the robot files gives
    the values linker-lab used before they were data (thumb flexion 4 + 0.4 × 1.86²,
    thumb yaw 24 − 20, fingers 4 + 0.4 × 0.89²), exactly."""
    motor = assets.motor(side)
    s = side[0]
    assert motor["shafts"] == tuple(
        f"hand_{s}h_{j}"
        for j in ("thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")
    )  # linker_sim.hand.SHAFTS order
    assert motor["motor_stiffness"] == 600.0 and motor["speed_limit"] == 1.0
    assert motor["advancing_cap"] == (5.0, 3.0, 5.0, 5.0, 5.0, 5.0)
    assert motor["output_damping"] == (40.0, 20.0, 40.0, 40.0, 40.0, 40.0)
    assert motor["viscosity"] == (20.0, 20.0, 4.2, 4.2, 4.2, 4.2)
    assert motor["passive_damping"] == (5.38384, 4.0, 4.31684, 4.31684, 4.31684, 4.31684)


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_controller_limits(side):
    import math

    limits = assets.controller(side)
    assert limits["joints"] == tuple(f"arm_AR5_5_08{side[0].upper()}_W4C4A6_joint_{i}" for i in range(1, 8))
    assert limits["max_velocity"] == (0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49)
    assert limits["max_acceleration"] == (10.0,) * 7 and limits["max_jerk"] == (40.0,) * 7
    corners = ((-23, 90), (23, 90), (60, 49), (60, -49), (23, -90), (-23, -90), (-60, -49), (-60, 49))
    assert limits["wrist_region"] == tuple((math.radians(a), math.radians(b)) for a, b in corners)
