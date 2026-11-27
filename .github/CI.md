# CI

Pull requests and pushes to `main` run static checks, Linux x86_64 core tests,
and a wheel installation check. A weekly schedule, version tags and manual
`workflow_dispatch` runs also cover macOS Apple Silicon and Linux ARM64, gallery
and viser acceptance, real-time diagnostics, the gallery's wheel contents and
copies from linker-lab. Python is 3.12, the only supported Python minor version.

Static checks install Ruff without the simulator dependencies. Runtime jobs use
`uv sync --locked` and the uv dependency cache. New pushes cancel older runs of
the same pull request. Action references are pinned to commits. Jobs have total
time limits; pytest checks also have per-test limits and CDP commands have a
30-second deadline. Results include slow-test durations and JUnit XML. Browser
screenshots are retained when the viewer tests produce them.

## Prerequisites

The reference fixtures must be committed under `tests/data/` (PR #9). A check fails
before dependency installation if any are missing or untracked. This workflow does
not copy data from a developer's working directory or download substitute fixtures.

Gallery tests require the files and checksums in the gallery manifests. When the
release is private, configure the repository Actions secret `GALLERY_READ_TOKEN`
with read access to its repository. The token is used only in extended runs, for
`gh release download`; the files are still checked against the manifest's SHA-256.
Without this secret the public URLs are used, and an unavailable file fails the job.

The copy check needs access to linker-bot/linker-lab at tag `v1.0.0`. For a private
repository, set `LINKER_LAB_READ_TOKEN` to a token with read access. The ordinary
workflow token is tried when that secret is absent; an inaccessible checkout fails
the job rather than silently skipping comparison. Update this tag together with
`tools/check_copies.py` when changing the recorded upstream release.

Extended jobs execute only on scheduled, manual or version-tag runs, not on
pull requests. Neither read token is passed to the PR jobs. A manual run uses the
selected branch; use a reviewed ref when running with repository secrets.

## Test groups

`tests/conftest.py` adds markers during collection. Plain `uv run pytest` still
runs the full suite. The core command is:

```sh
uv run pytest -m 'not gallery and not browser and not copies and not performance' \
  --timeout=120 --durations=15
```

`gallery` selects artifact-dependent gallery, lesson and documentation tests.
The manifest, downloader, CLI and lesson/document correspondence tests remain in
core. `browser` selects the viewer acceptance tests. `performance` selects checks
whose result includes a real-time requirement, so shared-runner load does not block
ordinary PR checks. `copies` selects the opt-in linker-lab comparison; the workflow
runs the standalone comparison directly without installing simulation dependencies.

The workflow uses viser on hosted runners. Studio's software rendering has not
completed reliably in local testing; Studio acceptance remains a separate check on
a machine with hardware WebGL. A green PR run does not establish full browser or
gallery acceptance. Chrome is required explicitly in the browser job, so its absence
does not turn the entire group into a passing skipped run.

## Wheel check and known blockers

The wheel job installs locked runtime dependencies, builds the wheel, replaces the
editable project installation with it, and invokes Python in isolated mode outside
the checkout. It checks asset access and CLI help without repeating simulation tests.
Extended runs also require `gallery.info("ycb-grasp")` to work from the wheel.

At introduction, the wheel omits gallery entries, and the gallery's public artifact
URL returns HTTP 404 in the review environment. The extended checks expose these
failures; this CI change does not fix the packaging or publish the policy files.
Core tests also remain blocked on a clean checkout until the fixtures in PR #9 land.

The initial job limits are safeguards, not measured performance targets. Use the
reported durations and cache hit rates from hosted runs before changing test
parallelism or adding more platform jobs. Browser jobs are deliberately sequential
within each process because they use local servers and share a CPU for rendering
and simulation.
