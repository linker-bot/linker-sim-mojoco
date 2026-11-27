"""CI prerequisites and wheel checks; no changes to the simulator's public API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ("drive_reference.npz", "gallery_reference.npz", "palm_reference.json")


def check_fixtures():
    for name in FIXTURES:
        path = f"tests/data/{name}"
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True
        )
        if tracked.returncode or not (ROOT / path).is_file():
            raise SystemExit(f"Missing committed test fixture: {path}. See PR #9.")


def fetch_gallery():
    from linker_sim import assets, gallery

    # urllib cannot use gh's authentication for private release assets. On trusted
    # scheduled/manual runs, GALLERY_READ_TOKEN may supply read access through gh.
    if os.environ.get("GH_TOKEN"):
        for name in gallery.entries():
            for artifact, spec in gallery.manifest(name)["artifacts"].items():
                path = gallery.artifact_path(name, artifact)
                if path.is_file() and assets.sha256(path) == spec["sha256"]:
                    continue
                url = urlparse(spec["url"])
                parts = url.path.strip("/").split("/")
                if url.hostname != "github.com" or len(parts) != 6 or parts[2:4] != ["releases", "download"]:
                    raise SystemExit(f"Not a GitHub release asset: {spec['url']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["gh", "release", "download", parts[4], "--repo", "/".join(parts[:2]),
                     "--pattern", parts[5], "--dir", str(path.parent), "--clobber"],
                    check=True, timeout=120,
                )
        # fetch() checks the manifest's SHA-256 even for authenticated downloads.
    for name in gallery.entries():
        gallery.fetch(name)


def wheel_smoke(gallery_required: bool):
    # Run outside the checkout, in isolated mode: an editable/source import would
    # otherwise hide missing wheel files. The workflow installs the wheel first.
    code = "from linker_sim import assets; assert assets.path('lw_v0.xml').is_file()"
    if gallery_required:
        code += "; from linker_sim import gallery; gallery.info('ycb-grasp')"
    with tempfile.TemporaryDirectory(prefix="linker-wheel-smoke-") as directory:
        subprocess.run([sys.executable, "-I", "-c", code], cwd=directory, check=True, timeout=60)
        subprocess.run(
            [sys.executable, "-I", "-m", "linker_sim.cli", "--help"],
            cwd=directory, check=True, timeout=60,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixtures", "gallery", "wheel"))
    parser.add_argument("--require-gallery", action="store_true")
    args = parser.parse_args()
    if args.command == "fixtures":
        check_fixtures()
    elif args.command == "gallery":
        fetch_gallery()
    else:
        wheel_smoke(args.require_gallery)
