from __future__ import annotations

import importlib.util
from pathlib import Path

DOCS_MAKE = Path(__file__).resolve().parents[1] / "docs" / "make.py"
SPEC = importlib.util.spec_from_file_location("kittensynth_docs_make", DOCS_MAKE)
assert SPEC is not None and SPEC.loader is not None
DOCS_MAKE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS_MAKE_MODULE)


def test_docs_build_uses_script_relative_paths_from_other_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    build_dir = tmp_path / "build"
    monkeypatch.setattr(DOCS_MAKE_MODULE, "BUILD_DIR", build_dir)
    commands = []
    monkeypatch.setattr(
        DOCS_MAKE_MODULE.subprocess,
        "run",
        lambda command, *, check: commands.append((command, check)),
    )

    assert DOCS_MAKE_MODULE.main(["html"]) == 0

    assert commands == [
        (
            [
                "sphinx-build",
                "-W",
                "-b",
                "html",
                str(DOCS_MAKE_MODULE.DOCS_DIR),
                str(build_dir / "html"),
            ],
            True,
        )
    ]
