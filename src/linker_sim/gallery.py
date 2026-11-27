"""The policy gallery: pretrained policies that run on LW-v0 (``linker-sim gallery``).

An entry is a folder in ``policy-gallery/`` of the repository:

    manifest.yaml   name, version, title, description, license, the linker-sim versions it
                    runs on (``linker_sim``, a version specifier) and its ``artifacts`` (file
                    name -> url and sha256)
    play.py         run(artifacts, viewer=, port=): opens the entry's scene in a viewer

``run`` checks the version range, downloads the artifacts into the user cache (checked
against their sha256), then calls the entry's ``play.run`` with the artifact paths and the
viewer. Each entry owns its scene, observation, action and episode code; the runner
interprets nothing else. An entry is for trying a policy interactively; measuring it
belongs to the repository it was trained in.

The cache is linker-lab's (``~/.cache/linker-lab/gallery/<entry>/<file>``): an entry that
both products ship uses the same files. To try an entry before its artifacts are
published, put each file at its cache path (:func:`artifact_path`); it is used if its
sha256 matches.

Copied from linker-lab v1.0.0:src/linker_lab/gallery.py (``linker_lab`` read as
``linker_sim``): the manifest format and its checks, the cache and the download
(``ROOT``, ``FIELDS``, ``entries``, ``manifest``, ``cache_dir``, ``artifact_path``, ``fetch``,
``load``); change them there first. New here: :func:`run` passes the viewer, and the texts
name linker-sim.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import urllib.request
from pathlib import Path

from linker_sim import __version__, assets

# The public API (docs/api.md); every other name may change in any release.
__all__ = ["entries", "info", "manifest", "supported", "run", "load", "fetch", "GalleryError"]

ROOT = Path(__file__).resolve().parents[2] / "policy-gallery"
FIELDS = (
    "name",
    "version",
    "title",
    "description",
    "license",
    "linker_sim",
    "artifacts",
)


class GalleryError(RuntimeError):
    """An entry is missing, malformed, not for this linker-sim, or its artifacts do not match."""


def entries() -> list[str]:
    """Names of the gallery's entries."""
    return sorted(p.parent.name for p in ROOT.glob("*/manifest.yaml"))


def manifest(name: str) -> dict:
    """An entry's manifest, checked: every field present, every artifact with a url and a
    sha256, a valid version specifier."""
    import yaml
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    file = ROOT / name / "manifest.yaml"
    if not file.is_file():
        raise GalleryError(f"no gallery entry {name!r}; entries: {', '.join(entries())}")
    data = yaml.safe_load(file.read_text())
    missing = [f for f in FIELDS if f not in data]
    if missing:
        raise GalleryError(f"{file}: missing {', '.join(missing)}")
    if data["name"] != name:
        raise GalleryError(f"{file}: name {data['name']!r} is not the folder name {name!r}")
    for artifact, spec in data["artifacts"].items():
        if not (
            isinstance(spec, dict) and spec.get("url") and len(str(spec.get("sha256", ""))) == 64
        ):
            raise GalleryError(f"{file}: artifact {artifact!r} needs a url and a sha256")
    try:
        SpecifierSet(data["linker_sim"])
    except InvalidSpecifier as error:
        raise GalleryError(f"{file}: linker_sim: {error}") from error
    if not (ROOT / name / "play.py").is_file():
        raise GalleryError(f"gallery entry {name!r} has no play.py")
    return data


def supported(data: dict) -> bool:
    """Whether this linker-sim's version is in the entry's range."""
    from packaging.specifiers import SpecifierSet

    return SpecifierSet(data["linker_sim"]).contains(__version__, prereleases=True)


def info(name: str) -> str:
    """A readable summary of an entry."""
    data = manifest(name)
    lines = [
        f"{data['name']} {data['version']}: {data['title']}",
        "",
        data["description"].strip(),
        "",
        f"License: {data['license'].strip()}",
        f"Runs on linker-sim {data['linker_sim']} (this is {__version__}"
        + ("" if supported(data) else ", NOT supported")
        + ")",
        "Artifacts:",
        *(f"  {k}: {v['url']}" for k, v in data["artifacts"].items()),
    ]
    return "\n".join(lines)


def cache_dir() -> Path:
    return assets.cache_dir().parent / "gallery"


def artifact_path(name: str, artifact: str) -> Path:
    return cache_dir() / name / artifact


def fetch(name: str) -> dict[str, Path]:
    """The entry's artifacts in the user cache, downloaded if missing, checked against their
    sha256."""
    data = manifest(name)
    paths = {}
    for artifact, spec in data["artifacts"].items():
        path = artifact_path(name, artifact)
        if not (path.is_file() and assets.sha256(path) == spec["sha256"]):
            _download(spec["url"], path)
            if assets.sha256(path) != spec["sha256"]:
                path.unlink()
                raise GalleryError(f"{spec['url']} does not match the sha256 in {name}'s manifest")
        paths[artifact] = path
    return paths


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"linker-sim: downloading {url}", file=sys.stderr)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
    except OSError as error:
        partial.unlink(missing_ok=True)
        hint = f"download it by hand to {destination}"
        parts = url.split("/")
        if url.startswith("https://github.com/") and parts[5:7] == ["releases", "download"]:
            # every file of the entry's release at once: owner/repo, tag
            hint += f", or all of the entry's files: gh release download {parts[7]} -R {parts[3]}/{parts[4]} -D {destination.parent}"
        raise GalleryError(f"could not download {url} ({error}); {hint}") from error
    partial.replace(destination)


def run(name: str, viewer: str = "studio", port: int = 0):
    """Open an entry's scene in ``viewer`` (``"studio"`` or ``"viser"``; ``port`` 0: the first
    free one from 8080; see the module docstring); returns when it is stopped (Ctrl+C)."""
    data = manifest(name)
    if not supported(data):
        raise GalleryError(f"{name} runs on linker-sim {data['linker_sim']}, this is {__version__}")
    return load(name).run(fetch(name), viewer=viewer, port=port)


def load(name: str):
    """An entry's ``play.py`` as a module (importing it prepares the engine)."""
    spec = importlib.util.spec_from_file_location(
        f"linker_sim_gallery_{name}", ROOT / name / "play.py"
    )
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    return play
