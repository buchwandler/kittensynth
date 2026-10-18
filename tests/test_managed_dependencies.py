from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from onnxvoice import OnnxVoice
from onnxvoice.errors import CatalogError, OptionalDependencyError

import kittensynth
from kittensynth import _onnxvoice
from kittensynth.errors import ManagedModelDependencyError


class _FailingManager:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def install(self, *_: object, **__: object) -> Any:
        raise self.error


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
