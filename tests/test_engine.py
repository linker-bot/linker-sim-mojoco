"""Pins, the doctor's checks and the command line."""

import subprocess
import sys

from linker_sim import engine


def run(code: str) -> str:
    """Run code in a fresh interpreter and return its last line of output."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_import_starts_nothing():
    out = run(
        "import sys, linker_sim, linker_sim.engine, linker_sim.cli, linker_sim.assets, linker_sim.physics, linker_sim.objects;"
        "print(sorted(m for m in ('mujoco', 'warp', 'torch', 'viser', 'mjviser') if m in sys.modules))"
    )
    assert out == "[]"


def test_installed_versions_match_pins():
    checks = engine.check_versions()
    names = {c.name for c in checks}
    assert {"mujoco", "warp-lang", "torch", "viser", "mjviser"} <= names
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_mujoco_is_the_pinned_engine():
    import mujoco

    assert mujoco.__version__ == "4.0.1"


def test_torch_is_the_cpu_build():
    import torch

    assert torch.version.cuda is None


def test_warp_runs_on_the_cpu():
    check = engine.check_warp()
    assert check.ok, check.detail


def test_one_simulated_second_holds_home_at_real_time():
    check = engine.check_simulation()
    assert check.ok, check.detail


def test_doctor_passes():
    out = subprocess.run(
        [sys.executable, "-m", "linker_sim.cli", "doctor"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "All checks passed." in out.stdout


def test_cli_help():
    out = subprocess.run(
        [sys.executable, "-m", "linker_sim.cli", "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "doctor" in out
