from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from kitteng2p.types import PhonemizeResult

from kittensynth import KittenVoice, ModelInferenceError, SynthesisConfig, VoiceLevelConfig
from kittensynth.voice_level import (
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
)


@dataclass
class FakeInference:
    audio: np.ndarray
    sample_rate: int = 24000


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, token_ids, **kwargs):
        self.calls.append((tuple(token_ids), kwargs))
        return FakeInference(np.array([[0.0, 0.25, -0.25]], dtype=np.float32))

    def close(self):
        self.closed = True


class FakeG2P:
    def __init__(self):
        self.closed = False

    def phonemize_prepared(self, text):
        return PhonemizeResult(
            text=text,
            raw_phonemes="həˈloʊ",
            phonemes="həˈloʊ",
            token_ids=(0, 1, 2, 10, 0),
            language="en-us",
        )

    def close(self):
        self.closed = True


def make_voices(path: Path):
    np.savez(
        path,
        **{"expr-voice-2-m": np.arange(40, dtype=np.float32).reshape(10, 4)},
    )


def make_model(tmp_path: Path, *, model_id: str | None = None):
    voices = tmp_path / "voices.npz"
    make_voices(voices)
    runtime = FakeRuntime()
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices,
        metadata={
            "voice_aliases": {"Jasper": "expr-voice-2-m"},
            "speed_priors": {"expr-voice-2-m": 0.8},
        },
        model_ref=f"kitten:{model_id}" if model_id else None,
        model_id=model_id,
        g2p=FakeG2P(),
    )
    return model, runtime


def test_synthesis_consumes_kitteng2p_ids_only(tmp_path):
    voices = tmp_path / "voices.npz"
    make_voices(voices)
    runtime = FakeRuntime()
    g2p = FakeG2P()
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices,
        metadata={
            "voice_aliases": {"Jasper": "expr-voice-2-m"},
            "speed_priors": {"expr-voice-2-m": 0.8},
        },
        model_ref="kitten:test",
        g2p=g2p,
    )

    result = model.synthesize_prepared("hello", voice="Jasper", speed=1.25)

    assert result.audio.dtype == np.float32
    ids, kwargs = runtime.calls[0]
    assert ids == (0, 1, 2, 10, 0)
    assert kwargs["style"].shape == (1, 4)
    assert kwargs["speed"] == 1.0
    assert result.metadata is not None
    assert result.metadata["phonemes"] == "həˈloʊ"

    model.close()
    assert runtime.closed
    assert g2p.closed is False  # injected g2p is caller-owned


def test_runtime_graph_failure_uses_model_inference_error(tmp_path):
    model, runtime = make_model(tmp_path)

    def fail_infer(*args, **kwargs):
        raise RuntimeError("graph execution failed")

    runtime.infer = fail_infer
    with pytest.raises(ModelInferenceError, match="inference failed") as error:
        model.synthesize_prepared("hello", voice="Jasper")

    assert isinstance(error.value.__cause__, RuntimeError)
    model.close()


def test_managed_calibration_key_uses_internal_voice_id(tmp_path):
    model, _ = make_model(tmp_path, model_id="nano-0.8-int8")

    assert str(model.calibration_key("Jasper")) == ("kitten:nano-0.8-int8:expr-voice-2-m")

    model.close()


def test_synthesis_applies_catalog_gain_and_records_application(tmp_path, monkeypatch):
    key = VoiceCalibrationKey("kitten", "nano-0.8-int8", "expr-voice-2-m")
    catalog = VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="test-v1",
        reference_lufs=-24.0,
        generated_with={},
        voices={key: VoiceLevelCalibration(gain_db=6.0)},
        revision="test-revision",
    )
    monkeypatch.setattr("kittensynth.voice_level.default_voice_calibration", lambda: catalog)
    model, _ = make_model(tmp_path, model_id="nano-0.8-int8")

    result = model.synthesize_prepared(
        "hello",
        voice="Jasper",
        config=SynthesisConfig(voice_level=VoiceLevelConfig(mode="calibrated")),
    )

    np.testing.assert_allclose(result.audio, np.array([0.0, 0.25, -0.25]) * (10.0 ** (6.0 / 20.0)))
    assert result.metadata is not None
    assert result.metadata["internal_voice"] == "expr-voice-2-m"
    assert result.metadata["model_id"] == "nano-0.8-int8"
    assert result.metadata["voice_level"] == {
        "mode": "calibrated",
        "applied": True,
        "gain_db": 6.0,
        "source": "catalog",
        "calibration_key": str(key),
        "reason": "a matching calibration catalog entry was selected",
        "catalog_revision": "test-revision",
    }
    assert model.last_voice_level_application is not None
    assert model.last_voice_level_application.key == key
    assert model.last_voice_level_application.gain_db == 6.0

    model.close()


def test_local_model_has_no_inferred_key_but_accepts_explicit_gain(tmp_path):
    model, _ = make_model(tmp_path)
    assert model.calibration_key("Jasper") is None

    calibrated = model.synthesize_prepared(
        "hello",
        voice="Jasper",
        config=SynthesisConfig(voice_level=VoiceLevelConfig(mode="calibrated")),
    )
    assert calibrated.metadata is not None
    assert calibrated.metadata["voice_level"]["source"] == "missing_identity"
    assert calibrated.metadata["voice_level"]["calibration_key"] is None
    np.testing.assert_array_equal(calibrated.audio, [0.0, 0.25, -0.25])

    overridden = model.synthesize_prepared(
        "hello",
        voice="Jasper",
        config=SynthesisConfig(voice_level=VoiceLevelConfig(gain_db=6.0)),
    )
    assert overridden.metadata is not None
    assert overridden.metadata["voice_level"]["source"] == "override"
    np.testing.assert_allclose(
        overridden.audio, np.array([0.0, 0.25, -0.25]) * (10.0 ** (6.0 / 20.0))
    )
    assert model.last_voice_level_application is not None
    assert model.last_voice_level_application.source == "override"

    model.close()


def test_explicit_config_speed_remains_authoritative(tmp_path):
    model, _ = make_model(tmp_path, model_id="nano-0.8-int8")

    result = model.synthesize_prepared(
        "hello",
        voice="Jasper",
        speed=1.25,
        config=SynthesisConfig(speed=2.0),
    )

    assert result.speed == 1.6
    model.close()


def test_runtime_close_failure_still_closes_owned_g2p(tmp_path, monkeypatch):
    voices_path = tmp_path / "voices.npz"
    make_voices(voices_path)
    runtime = FakeRuntime()
    owned_g2p = FakeG2P()
    monkeypatch.setattr("kittensynth.voice.KittenG2P", lambda: owned_g2p)

    def fail_close():
        runtime.closed = True
        raise RuntimeError("runtime close failed")

    runtime.close = fail_close
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices_path,
        metadata={},
        model_ref=None,
    )

    with pytest.raises(RuntimeError, match="runtime close failed"):
        model.close()

    assert runtime.closed
    assert owned_g2p.closed
    with pytest.raises(RuntimeError, match="KittenVoice is closed"):
        model.synthesize_prepared("hello")
    model.close()
