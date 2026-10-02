from __future__ import annotations

import pytest

from examples._output import artefact_dir, artefact_path


def test_environment_overrides_output_root(tmp_path, monkeypatch):
    output_root = tmp_path / "custom-output"
    monkeypatch.setenv("KITTENSYNTH_EXAMPLE_OUTPUT_DIR", str(output_root))

    assert artefact_dir() == output_root.resolve()
    assert output_root.is_dir()


def test_nested_artifact_creates_parent_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("KITTENSYNTH_EXAMPLE_OUTPUT_DIR", str(tmp_path / "output"))

    path = artefact_path("nested/deeper/sample.wav")

    assert path == (tmp_path / "output/nested/deeper/sample.wav").resolve()
    assert path.parent.is_dir()


def test_parent_traversal_is_rejected(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    monkeypatch.setenv("KITTENSYNTH_EXAMPLE_OUTPUT_DIR", str(output_root))

    with pytest.raises(ValueError, match="escapes"):
        artefact_path("../outside.wav")


def test_absolute_path_outside_root_is_rejected(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    monkeypatch.setenv("KITTENSYNTH_EXAMPLE_OUTPUT_DIR", str(output_root))

    with pytest.raises(ValueError, match="escapes"):
        artefact_path(tmp_path / "outside.wav")
