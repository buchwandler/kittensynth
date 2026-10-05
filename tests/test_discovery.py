from types import SimpleNamespace
from typing import Any, cast

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

    def install(self, *args, **kwargs):
        raise AssertionError("discovery must not install model artifacts")

    def open(self, *args, **kwargs):
        raise AssertionError("discovery must not open model artifacts")


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
            "voice_aliases": {"Jasper": "expr-voice-2-m", "Aria": "expr-voice-2-f"},
        },
        "voices": ("Jasper", "Aria"),
        "aliases": ("nano",),
        "sample_rate": 24000,
        "default_voice": None,
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
    manager_options = {}

    def make_manager(**kwargs):
        manager_options.update(kwargs)
        return manager

    monkeypatch.setattr(discovery, "_manager", make_manager)

    models = discovery.discover_models(
        language="en",
        offline=True,
        refresh=True,
        cache_dir="/tmp/models",
        catalog_url="https://catalog",
    )

    assert manager_options == {
        "cache_dir": "/tmp/models",
        "offline": True,
        "catalog_url": "https://catalog",
    }
    assert manager.calls == [
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
    assert model.metadata["source_revision"] == "abc123"
    assert model.runtime_available is True
    assert model.voice_ids == ("Jasper", "Aria")
    assert model.voices[0].gender == "male"
    assert model.voices[0].locale == "en-us"
    assert model.voices[1].gender == "female"
    assert model.voices[1].languages == ("en", "en-us")
    with pytest.raises(TypeError):
        cast(Any, model.metadata)["changed"] = True


def test_voice_aliases_get_default_english_metadata_without_gender_inference(monkeypatch):
    item = _kitten_item(
        voices=(),
        metadata={"voice_aliases": {"Hugo": "voice-m", "Luna": "voice-f"}},
    )
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))

    model = discovery.discover_models()[0]

    assert model.voice_ids == ("Hugo", "Luna")
    assert model.voices == (
        discovery.DescribedVoice("Hugo"),
        discovery.DescribedVoice("Luna"),
    )
    assert all(voice.gender == "unknown" for voice in model.voices)
    assert all(voice.language == "en" for voice in model.voices)
    assert all(voice.locale == "en" for voice in model.voices)
    assert all(voice.language_label == "English" for voice in model.voices)
    assert all(voice.languages == ("en",) for voice in model.voices)
    assert model.default_voice == "Hugo"


def test_default_voice_prefers_explicit_catalog_value(monkeypatch):
    item = _kitten_item(default_voice="Aria")
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))

    assert discovery.discover_models()[0].default_voice == "Aria"


def test_default_voice_prefers_jasper_then_first_alias(monkeypatch):
    jasper_item = _kitten_item(
        voices=("Aria", "Jasper"),
        metadata={"voice_aliases": {"Aria": "a", "Jasper": "j"}},
    )
    first_item = _kitten_item(
        id="first",
        voices=("Aria", "Luna"),
        metadata={"voice_aliases": {"Aria": "a", "Luna": "l"}},
    )
    monkeypatch.setattr(
        discovery, "_manager", lambda **kwargs: FakeManager([jasper_item, first_item], [])
    )

    models = discovery.discover_models()

    assert [model.default_voice for model in models] == ["Jasper", "Aria"]


def test_discover_models_uses_safe_sample_rate_fallback(monkeypatch):
    item = _kitten_item(sample_rate=0, metadata={"sample_rate": 22050})
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))
    assert discovery.discover_models()[0].sample_rate == 22050

    item = _kitten_item(sample_rate=0, metadata={})
    monkeypatch.setattr(discovery, "_manager", lambda **kwargs: FakeManager([item], []))
    assert discovery.discover_models()[0].sample_rate == 24000


def test_discover_models_wraps_catalog_failures(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "_manager",
        lambda **kwargs: FakeManager([], [], error=RuntimeError("catalog offline")),
    )

    with pytest.raises(kittensynth.CatalogUnavailableError, match="catalog offline"):
        discovery.discover_models(offline=True)


def test_discovery_api_is_exported_and_error_is_typed():
    for name in (
        "CatalogDiscoveryError",
        "DescribedVoice",
        "DiscoveredModel",
        "discover_models",
    ):
        assert name in kittensynth.__all__
        assert getattr(kittensynth, name) is getattr(discovery, name)
    assert issubclass(kittensynth.CatalogDiscoveryError, kittensynth.KittenSynthError)
    assert callable(kittensynth.runtime_identity)
