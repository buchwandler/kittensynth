from types import SimpleNamespace

import pytest

import kittensynth
from kittensynth import discovery


class FakeManager:
    def __init__(self, items, records, error=None):
        self.items = items
        self.records = records
        self.error = error
        self.calls = []

    def list(self, system, *, language=None, refresh=False):
        self.calls.append(("list", system, language, refresh))
        if self.error is not None:
            raise self.error
        return self.items

    def list_voices(self, system, *, language=None, refresh=False):
        self.calls.append(("list_voices", system, language, refresh))
        return self.records


def _kitten_item(**overrides):
    values = {
        "system": "kitten",
        "id": "nano-0.8-int8",
        "metadata": {
            "name": "Nano",
            "version": "0.8",
            "language": "en",
            "quality": "int8",
            "source_revision": "abc123",
        },
        "voices": ("Jasper", "Aria"),
        "aliases": ("nano",),
        "sample_rate": 24000,
        "default_voice": "Jasper",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _voice_record(item, voice_id, metadata, languages=("en",)):
    return SimpleNamespace(
        catalog_item=item,
        voice_id=voice_id,
        metadata=metadata,
        languages=languages,
    )


def test_discover_models_maps_catalog_items_and_voice_metadata(monkeypatch):
    item = _kitten_item()
    records = (
        _voice_record(
            item,
            "Jasper",
            SimpleNamespace(gender="male", language="en", locale="en-us", language_label="English"),
        ),
        _voice_record(
            item,
            "Aria",
            {"gender": "female", "language": "en", "locale": "en-us", "language_label": "English"},
            languages=("en", "en-us"),
        ),
        _voice_record(
            _kitten_item(id="other-model"),
            "Ignored",
            {"gender": "unknown", "language": "en", "locale": "en", "language_label": "English"},
        ),
    )
    manager = FakeManager([item, _kitten_item(system="piper", id="piper-item")], list(records))
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: manager)

    models = discovery.discover_models(language="en", offline=True, refresh=True)

    assert manager.calls[:2] == [
        ("list", "kitten", "en", True),
        ("list_voices", "kitten", "en", False),
    ]
    assert [model.id for model in models] == ["nano-0.8-int8"]
    model = models[0]
    assert model.display_name == "Nano"
    assert model.version == "0.8"
    assert model.language == "en"
    assert model.quality == "int8"
    assert model.sample_rate == 24000
    assert model.aliases == ("nano",)
    assert model.default_voice == "Jasper"
    assert model.source_revision == "abc123"
    assert model.runtime_available is True
    assert model.voice_ids == ("Jasper", "Aria")
    assert model.voices[0].gender == "male"
    assert model.voices[0].locale == "en-us"
    assert model.voices[1].gender == "female"
    assert model.voices[1].languages == ("en", "en-us")


def test_discover_models_falls_back_to_item_voice_ids(monkeypatch):
    item = _kitten_item()
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))

    models = discovery.discover_models()

    assert models[0].voices == (
        discovery.DescribedVoice(id="Jasper"),
        discovery.DescribedVoice(id="Aria"),
    )


def test_discover_models_normalizes_invalid_sample_rate(monkeypatch):
    item = _kitten_item(sample_rate=0)
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))

    assert discovery.discover_models()[0].sample_rate is None


def test_discover_models_wraps_failures_in_catalog_discovery_error(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "_manager",
        lambda **kwargs: FakeManager([], [], error=RuntimeError("catalog offline")),
    )

    with pytest.raises(discovery.CatalogDiscoveryError, match="catalog offline"):
        discovery.discover_models(offline=True)


def test_runtime_identity_is_opaque_and_carries_revisions(monkeypatch):
    monkeypatch.setattr(discovery, "_distribution_version_safe", lambda name: f"{name}-1.0")

    identity = discovery.runtime_identity()
    assert identity == {
        "engine_version": "kittensynth-1.0",
        "runtime_revision": "onnxvoice-1.0",
        "catalog_revision": None,
        "model_revision": None,
    }

    item = _kitten_item()
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))
    model = discovery.discover_models()[0]
    identity = discovery.runtime_identity(model)
    assert identity["catalog_revision"] == "abc123"
    assert identity["model_revision"] == "0.8"


def test_discovery_api_is_exported_and_error_is_typed():
    for name in (
        "CatalogDiscoveryError",
        "DescribedVoice",
        "DiscoveredModel",
        "discover_models",
        "runtime_identity",
    ):
        assert name in kittensynth.__all__
        assert getattr(kittensynth, name) is getattr(discovery, name)
    assert issubclass(kittensynth.CatalogDiscoveryError, kittensynth.KittenSynthError)
