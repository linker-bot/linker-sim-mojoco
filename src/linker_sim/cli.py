"""The ``linker-sim`` command."""

from __future__ import annotations

import argparse
import platform
import sys


def doctor() -> int:
    """Check the installation: versions, Warp and torch on the CPU, and one simulated second."""
    import linker_sim
    from linker_sim import engine

    checks = [*engine.check_versions(), engine.check_warp(), engine.check_torch()]
    if all(c.ok for c in checks):
        checks.append(engine.check_simulation())
    print(f"\nlinker-sim {linker_sim.__version__} doctor ({platform.system()} {platform.machine()},"
          f" Python {platform.python_version()})")
    for c in checks:
        print(f"  [{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    ok = all(c.ok for c in checks)
    print("All checks passed." if ok else "Some checks failed; see above.")
    return 0 if ok else 1


def play(viewer: str, port: int) -> int:
    """Drive LW-v0 by hand in a viewer (the playground); open the address it prints."""
    from linker_sim import playground

    playground.run(viewer, port)
    return 0


def gallery(command: str, name: str | None, viewer: str = "studio", port: int = 0) -> int:
    """Pretrained policies on LW-v0: list the entries, show one, or open one in a viewer."""
    from linker_sim import gallery as g

    if command == "list":
        for entry in g.entries():
            data = g.manifest(entry)
            supported = "" if g.supported(data) else "  (not supported)"
            print(f"{entry:16s} {data['title']}{supported}")
    elif command == "info":
        print(g.info(name))
    else:
        g.run(name, viewer=viewer, port=port)
    return 0


def _viewer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--viewer", choices=("studio", "viser"), default="studio",
        help="MuJoCo Studio (default) or viser; both are web pages on this computer",
    )  # fmt: skip
    parser.add_argument("--port", type=int, default=0, help="port of the page (default: first free from 8080)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linker-sim", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help=doctor.__doc__)
    _viewer_options(commands.add_parser("play", help=play.__doc__))
    gallery_parser = commands.add_parser("gallery", help=gallery.__doc__)
    gallery_commands = gallery_parser.add_subparsers(dest="gallery_command", required=True)
    gallery_commands.add_parser("list", help="the gallery's entries")
    gallery_commands.add_parser("info", help="an entry's description and license").add_argument("entry")
    run_parser = gallery_commands.add_parser("run", help="open an entry in a viewer")
    run_parser.add_argument("entry")
    _viewer_options(run_parser)
    args = parser.parse_args(argv)
    from linker_sim.gallery import GalleryError

    try:
        return _run(args)
    except GalleryError as error:  # e.g. a file that could not be downloaded
        print(f"linker-sim: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        return doctor()
    if args.command == "play":
        return play(args.viewer, args.port)
    if args.command == "gallery":
        if args.gallery_command == "run":
            return gallery("run", args.entry, args.viewer, args.port)
        return gallery(args.gallery_command, getattr(args, "entry", None))
    return 2


if __name__ == "__main__":
    sys.exit(main())
