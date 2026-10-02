from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from kittensynth import KittenVoice, SynthesisConfig
from kittensynth.errors import InvalidSynthesisConfigError
from kittensynth.voice_level import (
    CalibrationDataError,
    VoiceCalibrationKey,
    VoiceLevelConfig,
    apply_voice_level_calibration,
    load_voice_calibration,
)


def catalog_data(voices: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": 1,
        "method": "bs1770",
        "corpus": "prepared-test-v1",
        "reference_lufs": -24.0,
        "generated_with": {"audiosig": "0.1.6"},
        "voices": voices or {},
    }


def write_catalog(tmp_path: Path, data: dict[str, object], name: str = "catalog.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_voice_level_config_defaults_to_off():
    assert VoiceLevelConfig().mode == "off"


@pytest.mark.parametrize("mode", ["automatic", "", None, 1])
def test_voice_level_config_rejects_invalid_mode(mode):
    with pytest.raises(InvalidSynthesisConfigError):
        VoiceLevelConfig(mode=mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("gain_db", [float("nan"), float("inf"), -float("inf"), True])
def test_voice_level_config_rejects_non_finite_or_boolean_gain(gain_db):
    with pytest.raises(InvalidSynthesisConfigError):
        VoiceLevelConfig(gain_db=gain_db)


def test_valid_catalog_loads_with_deterministic_revision(tmp_path):
    key = "kitten:nano-0.8-int8:expr-voice-2-m"
    data = catalog_data({key: {"gain_db": -1.5, "samples": 9}})
    path = write_catalog(tmp_path, data)

    catalog = load_voice_calibration(path)
    second = load_voice_calibration(write_catalog(tmp_path, data, "second.json"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert catalog.revision == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert catalog.revision == second.revision
    assert catalog.voices[VoiceCalibrationKey.parse(key)].gain_db == -1.5


def test_catalog_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":1,"method":"bs1770","corpus":"x",'
        '"reference_lufs":-24,"generated_with":{},"voices":{},"voices":{}}',
        encoding="utf-8",
    )

    with pytest.raises(CalibrationDataError, match="duplicate JSON object key"):
        load_voice_calibration(path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update(extra=True), "unknown field"),
        (lambda data: data.update(schema=2), "unsupported calibration schema"),
        (lambda data: data.update(schema=True), "unsupported calibration schema"),
        (lambda data: data.update(method="peak"), "unsupported calibration method"),
        (lambda data: data.update(corpus=""), "corpus must be a non-empty string"),
        (lambda data: data.update(generated_with={"audiosig": 1}), "string mapping"),
        (lambda data: data.update(reference_lufs=float("nan")), "non-finite JSON number"),
    ],
)
def test_catalog_rejects_invalid_top_level_data(tmp_path, mutate, message):
    data = catalog_data()
    mutate(data)

    with pytest.raises(CalibrationDataError, match=message):
        load_voice_calibration(write_catalog(tmp_path, data))


def test_catalog_rejects_unknown_record_fields(tmp_path):
    data = catalog_data({"kitten:model:voice": {"gain_db": 0.5, "unexpected": "value"}})

    with pytest.raises(CalibrationDataError, match="unknown field"):
        load_voice_calibration(write_catalog(tmp_path, data))


@pytest.mark.parametrize(
    "record, message",
    [
        ({"gain_db": float("inf")}, "non-finite JSON number"),
        ({"gain_db": 1e309}, "non-finite JSON number"),
        ({"gain_db": 0, "samples": 0}, "positive integer"),
        ({"gain_db": 0, "method": "other"}, "unsupported method"),
        ({"gain_db": 0, "corpus_version": 12}, "corpus_version must be a string"),
    ],
)
def test_catalog_rejects_invalid_record_values(tmp_path, record, message):
    data = catalog_data({"kitten:model:voice": record})

    with pytest.raises(CalibrationDataError, match=message):
        load_voice_calibration(write_catalog(tmp_path, data))


def test_catalog_rejects_overflowing_json_number(tmp_path):
    data = catalog_data({"kitten:model:voice": {"gain_db": 0.0}})
    raw = json.dumps(data).replace('"gain_db": 0.0', '"gain_db": 1e309')
    path = tmp_path / "overflow.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(CalibrationDataError, match="must be finite"):
        load_voice_calibration(path)


def test_catalog_rejects_ambiguous_or_non_kitten_key(tmp_path):
    for index, key in enumerate(
        ("kitten:model:voice:extra", "piper:model:voice", "kitten:bad:model:id")
    ):
        path = write_catalog(
            tmp_path, catalog_data({key: {"gain_db": 0.5}}), f"invalid-{index}.json"
        )
        with pytest.raises(CalibrationDataError):
            load_voice_calibration(path)


def test_off_mode_returns_equivalent_audio_and_off_source():
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(audio, VoiceLevelConfig(), None)

    assert np.array_equal(output, audio)
    assert output is not audio
    assert application.source == "off"
    assert application.gain_db == 0.0
    assert not application.applied


def test_explicit_override_is_applied_even_when_mode_is_off():
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="off", gain_db=6.0), None
    )

    np.testing.assert_allclose(output, audio * (10.0 ** (6.0 / 20.0)))
    assert application.source == "override"
    assert application.applied
    assert application.gain_db == 6.0


def test_matching_catalog_identity_is_applied(tmp_path):
    key = VoiceCalibrationKey("kitten", "nano-0.8-int8", "expr-voice-2-m")
    catalog = load_voice_calibration(
        write_catalog(tmp_path, catalog_data({str(key): {"gain_db": -3.0}}))
    )
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=catalog
    )

    np.testing.assert_allclose(output, audio * (10.0 ** (-3.0 / 20.0)))
    assert application.source == "catalog"
    assert application.key == key
    assert application.catalog_revision == catalog.revision


def test_zero_db_override_is_not_reported_as_applied():
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(audio, VoiceLevelConfig(gain_db=0.0), None)

    np.testing.assert_array_equal(output, audio)
    assert application.source == "override"
    assert application.gain_db == 0.0
    assert not application.applied


def test_zero_db_catalog_is_not_reported_as_applied(tmp_path):
    key = VoiceCalibrationKey("kitten", "nano-0.8-int8", "expr-voice-2-m")
    catalog = load_voice_calibration(
        write_catalog(tmp_path, catalog_data({str(key): {"gain_db": 0.0}}))
    )
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=catalog
    )

    np.testing.assert_array_equal(output, audio)
    assert application.source == "catalog"
    assert application.gain_db == 0.0
    assert not application.applied


def test_nonzero_gain_is_not_applied_when_audio_is_unchanged():
    audio = np.zeros(2, dtype=np.float32)

    output, application = apply_voice_level_calibration(audio, VoiceLevelConfig(gain_db=6.0), None)

    np.testing.assert_array_equal(output, audio)
    assert application.gain_db == 6.0
    assert not application.applied


def test_missing_managed_identity_leaves_audio_unchanged():
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), None
    )

    np.testing.assert_array_equal(output, audio)
    assert application.source == "missing_identity"
    assert "stable managed" in application.reason


def test_missing_catalog_record_leaves_audio_unchanged(tmp_path):
    key = VoiceCalibrationKey("kitten", "nano-0.8-int8", "expr-voice-2-m")
    catalog = load_voice_calibration(write_catalog(tmp_path, catalog_data()))
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=catalog
    )

    np.testing.assert_array_equal(output, audio)
    assert application.source == "missing_calibration"
    assert application.catalog_revision == catalog.revision
    assert math.isfinite(application.gain_db)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KITTENSYNTH_RUN_INTEGRATION") != "1",
    reason="set KITTENSYNTH_RUN_INTEGRATION=1 to use cached model assets",
)
def test_managed_calibrated_voice_saves_valid_wav(tmp_path: Path):

    config = SynthesisConfig(
        speed=1.0,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    output = tmp_path / "calibrated-jasper.wav"
    with KittenVoice.from_pretrained("nano-0.8-int8", offline=True) as model:
        result = model.synthesize_prepared(
            "Prepared speech demonstrates calibrated Kitten voice level.",
            voice="Jasper",
            config=config,
        )
        application = model.last_voice_level_application
        assert application is not None
        assert application.source == "catalog"
        assert application.key == VoiceCalibrationKey("kitten", "nano-0.8-int8", "expr-voice-2-m")
        assert result.metadata is not None
        assert result.metadata["voice_level"]["applied"] is True
        result.save_wav(output)

    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() > 0
        assert wav.getnframes() > 0
