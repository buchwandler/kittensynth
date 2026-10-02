#!/usr/bin/env python3
"""Build the KittenSynth documentation with warning-as-error checks."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = DOCS_DIR
BUILD_DIR = DOCS_DIR / "_build"
VALID_TARGETS = {
    "html",
    "dirhtml",
    "latex",
    "latexpdf",
    "text",
    "man",
    "changes",
    "linkcheck",
    "doctest",
    "all",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run Sphinx for a target, independent of the caller's working directory."""
    args = list(sys.argv[1:] if argv is None else argv)
    target = args[0] if args else "html"

    if target == "clean":
        if BUILD_DIR.exists():
            print(f"Cleaning {BUILD_DIR}...")
            shutil.rmtree(BUILD_DIR)
        return 0
    if target == "help":
        print(__doc__)
        print("Targets: " + ", ".join(sorted(VALID_TARGETS | {"clean", "help"})))
        return 0
    if target not in VALID_TARGETS:
        print(f"Unknown target: {target}")
        print("Use 'help' target for help")
        return 1

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    formats = ("html", "dirhtml", "latex") if target == "all" else (target,)
    for format_name in formats:
        print(f"Building {format_name} documentation...")
        if format_name == "latexpdf":
            command = [
                "sphinx-build",
                "-W",
                "-M",
                format_name,
                str(SOURCE_DIR),
                str(BUILD_DIR),
            ]
        else:
            command = [
                "sphinx-build",
                "-W",
                "-b",
                format_name,
                str(SOURCE_DIR),
                str(BUILD_DIR / format_name),
            ]
        subprocess.run(command, check=True)

    print(f"Build finished. Documentation is in {BUILD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
