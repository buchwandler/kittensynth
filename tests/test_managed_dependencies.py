from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from onnxvoice import OnnxVoice
from onnxvoice.errors import (
    AssetDownloadError,
    AssetTransportError,
    CatalogError,
    OptionalDependencyError,
)

import kittensynth
from kittensynth import _onnxvoice
from kittensynth.errors import ManagedModelDependencyError


class _FailingManager:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def install(self, *_: object, **__: object) -> Any:
        raise self.error


class _InstallAndOpenManager:
    def __init__(self, installation: Any, open_error: Exception | None = None) -> None:
        self.installation = installation
        self.open_error = open_error

    def install(self, *_: object, **__: object) -> Any:
        return self.installation

    def open(self, *_: object, **__: object) -> Any:
        if self.open_error is not None:
            raise self.open_error
        return object()


def _make_installation(tmp_path: Path) -> Any:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.npz"
    model_path.touch()
    voices_path.touch()
    artifacts = {
        "model": SimpleNamespace(path=model_path),
        "voices": SimpleNamespace(path=voices_path),
    }
    return SimpleNamespace(
        system="kitten",
        id="micro-0.8",
        ref="kitten:micro-0.8",
        sample_rate=24000,
        metadata={},
        artifact=lambda role: artifacts[role],
    )


def test_missing_hub_transport_maps_to_public_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OptionalDependencyError("private-token=do-not-leak")
    manager = _FailingManager(original)
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: manager)

    with pytest.raises(ManagedModelDependencyError) as caught:
        _onnxvoice.install_pretrained_model("nano-0.8-int8")

    assert caught.value.__cause__ is original
    assert "huggingface" in str(caught.value)
    assert "private-token" not in str(caught.value)
    assert "Pocket" not in str(caught.value)
    assert kittensynth.ManagedModelDependencyError is ManagedModelDependencyError


def test_asset_download_error_maps_to_public_error_without_leaking_low_level_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_level = RuntimeError("private-token=do-not-leak")
    original = AssetDownloadError(
        "Could not download Hugging Face asset kitten:micro-0.8 (RuntimeError)"
    )
    original.__cause__ = low_level
    manager = _FailingManager(original)
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: manager)

    with pytest.raises(kittensynth.ModelDownloadError) as caught:
        _onnxvoice.install_pretrained_model("micro-0.8")

    assert caught.value.__cause__ is original
    assert caught.value.__cause__.__cause__ is low_level
    assert "Could not download Hugging Face asset kitten:micro-0.8" in str(caught.value)
    assert "private-token" not in str(caught.value)


def test_asset_transport_subclass_maps_and_preserves_transport_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AssetTransportError(
        "Could not download Hugging Face asset kitten:micro-0.8 (ValueError); "
        "HTTP-mode retry was attempted and failed; direct HTTPS fallback was attempted and failed.",
        repository="example/kitten",
        revision="main",
        path="model.onnx",
        stage="xet_fallback",
        native_error_type="URLError",
        fallback_attempted=True,
        fallback_failed=True,
    )
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: _FailingManager(original))

    with pytest.raises(kittensynth.ModelDownloadError) as caught:
        _onnxvoice.install_pretrained_model("micro-0.8")

    assert caught.value.__cause__ is original
    assert isinstance(caught.value.__cause__, AssetTransportError)
    assert caught.value.__cause__.stage == "xet_fallback"
    assert "direct HTTPS fallback was attempted and failed" in str(caught.value)


def test_successful_install_resolves_model_and_voice_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installation = _make_installation(tmp_path)
    manager = _InstallAndOpenManager(installation)
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: manager)

    resolved = _onnxvoice.install_pretrained_model("micro-0.8")

    assert resolved.installation is installation
    assert resolved.model_id == "micro-0.8"
    assert resolved.model_path == installation.artifact("model").path
    assert resolved.voices_path == installation.artifact("voices").path


def test_open_failure_after_install_is_not_classified_as_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installation = _make_installation(tmp_path)
    original = RuntimeError("provider initialization failed")
    manager = _InstallAndOpenManager(installation, open_error=original)
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: manager)

    with pytest.raises(RuntimeError) as caught:
        kittensynth.KittenVoice.from_pretrained("micro-0.8")

    assert caught.value is original
    assert not isinstance(caught.value, kittensynth.ModelDownloadError)


def test_catalog_errors_are_not_reclassified_as_missing_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CatalogError("catalog unavailable")
    manager = _FailingManager(original)
    monkeypatch.setattr(_onnxvoice, "_manager", lambda **_: manager)

    with pytest.raises(CatalogError) as caught:
        _onnxvoice.install_pretrained_model("nano-0.8-int8")

    assert caught.value is original


def _assert_managed_artifacts_exist(cache_dir: Path) -> None:
    installations = OnnxVoice(cache_dir=cache_dir).installed(system="kitten")
    installation = next(item for item in installations if item.id == "nano-0.8-int8")
    for role in ("model", "voices"):
        assert installation.artifact(role).path.is_file()


@pytest.mark.integration
def test_managed_cold_download_and_offline_cache_reuse(tmp_path: Path) -> None:
    if os.environ.get("KITTENSYNTH_RUN_MANAGED_INTEGRATION") != "1":
        pytest.skip("set KITTENSYNTH_RUN_MANAGED_INTEGRATION=1 to download the public model")

    cache_dir = tmp_path / "fresh-cache"
    assert not cache_dir.exists()

    with kittensynth.KittenVoice.from_pretrained(
        "nano-0.8-int8",
        cache_dir=cache_dir,
    ) as model:
        measured = model.measure_prepared("Hello from KittenSynth.")
        assert measured.token_count > 0
        assert measured.fits
        result = model.synthesize_prepared("Hello from KittenSynth.", voice="Jasper")
        assert result.audio.size > 0
        assert result.sample_rate == 24000

    _assert_managed_artifacts_exist(cache_dir)

    with kittensynth.KittenVoice.from_pretrained(
        "nano-0.8-int8",
        cache_dir=cache_dir,
        offline=True,
    ) as model:
        measured = model.measure_prepared("Offline cache reuse.")
        assert measured.fits
        result = model.synthesize_prepared("Offline cache reuse.", voice="Jasper")
        assert result.audio.size > 0
