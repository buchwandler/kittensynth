from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kitteng2p import KittenG2P
from kitteng2p.types import PhonemizeResult

from kittensynth import (
    EmptyTextError,
    KittenVoice,
    ModelInferenceError,
    SynthesisConfig,
    SynthesisInputTooLongError,
    VoiceLevelConfig,
)
from kittensynth.capacity import capacity_for_model
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


class InputSensitiveFakeG2P(FakeG2P):
    def phonemize_prepared(self, text):
        token_ids = tuple(range(len(text.split()) + 1))
        return PhonemizeResult(
            text=text,
            raw_phonemes="input-sensitive",
            phonemes="input-sensitive",
            token_ids=token_ids,
            language="en-us",
        )


class TextPhonemeBackend:
    def phonemize(self, text: str, *, language: str) -> str:
        return text

    def close(self):
        pass


class FixedTokenFakeG2P:
    def __init__(self, token_count: int):
        self.token_ids = (0, *(1 for _ in range(token_count - 3)), 10, 0)

    def phonemize_prepared(self, text: str):
        return PhonemizeResult(
            text=text,
            raw_phonemes="fixed",
            phonemes="fixed",
            token_ids=self.token_ids,
            language="en-us",
        )


def make_voices(path: Path):
    np.savez(
        path,
        **{"expr-voice-2-m": np.arange(40, dtype=np.float32).reshape(10, 4)},
    )


def make_model(
    tmp_path: Path,
    *,
    model_id: str | None = None,
    g2p: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    voices = tmp_path / "voices.npz"
    make_voices(voices)
    runtime = FakeRuntime()
    model_metadata = {
        "voice_aliases": {"Jasper": "expr-voice-2-m"},
        "speed_priors": {"expr-voice-2-m": 0.8},
    }
    if metadata is not None:
        model_metadata.update(metadata)
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices,
        metadata=model_metadata,
        model_ref=f"kitten:{model_id}" if model_id else None,
        model_id=model_id,
        g2p=g2p if g2p is not None else FakeG2P(),
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


def test_measure_prepared_returns_exact_token_count_without_inference(tmp_path):
    model, runtime = make_model(tmp_path)

    measure = model.measure_prepared("hello")

    assert measure.token_count == 5
    assert measure.max_tokens == 512
    assert measure.fits is True
    assert runtime.calls == []
    model.close()


def test_measurement_and_synthesis_share_input_sensitive_frontend(tmp_path):
    voices_path = tmp_path / "voices.npz"
    make_voices(voices_path)
    runtime = FakeRuntime()
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices_path,
        metadata={"voice_aliases": {"Jasper": "expr-voice-2-m"}},
        model_ref=None,
        g2p=InputSensitiveFakeG2P(),
    )
    text = "hello there"

    measure = model.measure_prepared(text)
    assert measure.token_count == 3
    assert measure.token_count != len(text)
    model.synthesize_prepared(text)
    assert measure.token_count == len(runtime.calls[-1][0])
    model.close()


def test_measurement_succeeds_when_inference_would_fail(tmp_path):
    model, runtime = make_model(tmp_path)

    def fail_infer(*args, **kwargs):
        raise AssertionError("measurement must not call inference")

    runtime.infer = fail_infer
    assert model.measure_prepared("hello").token_count == 5
    model.close()


@pytest.mark.parametrize("text", ["", "   "])
def test_measurement_and_synthesis_reject_empty_text_consistently(tmp_path, text):
    model, runtime = make_model(tmp_path)

    with pytest.raises(EmptyTextError, match="speakable content"):
        model.measure_prepared(text)
    with pytest.raises(EmptyTextError, match="speakable content"):
        model.synthesize_prepared(text)
    assert runtime.calls == []
    model.close()


def test_measurement_rejects_closed_voice(tmp_path):
    model, _ = make_model(tmp_path)
    model.close()

    with pytest.raises(RuntimeError, match="closed"):
        model.measure_prepared("hello")


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


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "  extra   spaces\tand\nlines  ",
        "2026 costs $5.00.",
        "This is a normal sentence.",
    ],
)
def test_measurement_matches_model_input_from_kitteng2p_for_text_forms(tmp_path, text):
    g2p = KittenG2P(backend=TextPhonemeBackend())
    model, runtime = make_model(
        tmp_path,
        model_id="nano-0.8-int8",
        g2p=g2p,
        metadata={"source_revision": "revision-1"},
    )

    measured = model.measure_prepared(text)
    assert runtime.calls == []

    model.synthesize_prepared(text)
    model_input = runtime.calls[-1][0]

    assert measured.token_count == len(model_input)
    assert measured.max_tokens == 512
    assert measured.fits is (len(model_input) <= measured.max_tokens)
    assert model_input[0] == 0
    assert model_input[-2:] == (10, 0)
    model.close()
    g2p.close()


@pytest.mark.parametrize(
    ("token_count", "fits"),
    [(511, True), (512, True), (513, False)],
)
def test_synthesis_enforces_exact_framed_input_boundaries(tmp_path, token_count, fits):
    g2p = FixedTokenFakeG2P(token_count)
    model, runtime = make_model(tmp_path, model_id="nano-0.8-int8", g2p=g2p)

    measured = model.measure_prepared("boundary")
    assert measured.token_count == token_count
    assert measured.max_tokens == 512
    assert measured.fits is fits

    if fits:
        model.synthesize_prepared("boundary")
        assert runtime.calls[-1][0] == g2p.token_ids
    else:
        with pytest.raises(SynthesisInputTooLongError) as caught:
            model.synthesize_prepared("boundary")
        assert caught.value.amount == 513
        assert caught.value.maximum == 512
        assert caught.value.unit == "model_tokens"
        assert runtime.calls == []
    model.close()


def test_measurement_policy_revision_is_stable_for_model_and_frontend(tmp_path):
    model, _ = make_model(
        tmp_path,
        model_id="nano-0.8-int8",
        metadata={"source_revision": "revision-1"},
    )

    measured = model.measure_prepared("hello")
    expected = capacity_for_model("nano-0.8-int8", "revision-1")

    assert measured.metadata["capacity_policy_revision"] == expected.policy_revision
    assert (
        expected.policy_revision
        == capacity_for_model("nano-0.8-int8", "revision-1").policy_revision
    )
    assert (
        expected.policy_revision
        != capacity_for_model("nano-0.8-int8", "revision-2").policy_revision
    )
    model.close()
