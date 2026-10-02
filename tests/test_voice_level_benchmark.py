from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks.promote_voice_calibration import PRODUCTION_CATALOG, build_candidate, main
from benchmarks.voice_level_benchmark import (
    aggregate_measurements,
    build_report,
    expand_identities,
    load_policy,
    load_stimuli,
    measure_identities,
)
from kittensynth.voice_bank import DEFAULT_VOICE_ALIASES
from kittensynth.voice_level import VoiceCalibrationKey, load_voice_calibration


class FakeVoiceBank:
    def resolve(self, alias: str) -> str:
        return DEFAULT_VOICE_ALIASES.get(alias, alias)


class FakeModel:
    def __init__(self) -> None:
        self.voice_bank = FakeVoiceBank()
        self.calls: list[dict[str, Any]] = []

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
        self.calls.append({"text": text, "voice": voice, "config": config})
        return SimpleNamespace(
            audio=np.array([0.0, 0.2, -0.2], dtype=np.float32),
            sample_rate=24000,
            duration_seconds=0.1,
            speed=1.0,
        )


def identity(alias: str, internal_voice: str) -> dict[str, Any]:
    key = str(VoiceCalibrationKey("kitten", "nano-0.8-int8", internal_voice))
    return {
        "model_source": "kitten",
        "requested_model": "nano-0.8-int8",
        "model_id": "nano-0.8-int8",
        "model_ref": "kitten:nano-0.8-int8",
        "voice_alias": alias,
        "internal_voice": internal_voice,
        "calibration_key": key,
    }


def stimuli() -> list[dict[str, str]]:
    return [
        {"id": "short", "text": "A calm voice speaks clearly."},
        {"id": "medium", "text": "Prepared speech gives the synthesizer enough material."},
        {
            "id": "long",
            "text": "A longer prepared sentence compares relative levels across all voices.",
        },
    ]


def synthetic_measurements(
    entry: dict[str, Any],
    stimulus_values: dict[str, list[float]],
    *,
    raw_peak: float = 0.2,
) -> list[dict[str, Any]]:
    rows = []
    for stimulus in stimuli():
        values = stimulus_values[stimulus["id"]]
        for repeat, integrated_lufs in enumerate(values):
            rows.append(
                {
                    **entry,
                    "stimulus_id": stimulus["id"],
                    "repeat": repeat,
                    "requested_speed": 1.0,
                    "effective_speed": 1.0,
                    "sample_rate": 24000,
                    "duration_seconds": 1.0,
                    "integrated_lufs": integrated_lufs,
                    "raw_peak": raw_peak,
                    "calibration_mode": "off",
                }
            )
    return rows


def values_around(median: float) -> dict[str, list[float]]:
    return {
        "short": [median - 0.1, median, median + 0.1],
        "medium": [median - 0.1, median, median + 0.1],
        "long": [median - 0.1, median, median + 0.1],
    }


def report_for(
    entries: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    *,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_report(
        entries,
        measurements,
        failures or [],
        stimuli=stimuli(),
        policy=load_policy(),
        language="en-us",
    )


def test_default_stimuli_cover_short_medium_and_long_prepared_text():
    document = load_stimuli()

    assert [stimulus["id"] for stimulus in document["stimuli"]] == [
        "short",
        "medium",
        "long",
    ]
    assert all(stimulus["text"].strip() for stimulus in document["stimuli"])


def test_identity_expansion_covers_every_default_internal_voice():
    aliases = tuple(DEFAULT_VOICE_ALIASES)
    entries = expand_identities(
        aliases,
        FakeVoiceBank(),
        requested_model="nano-0.8-int8",
        model_id="nano-0.8-int8",
        model_ref="kitten:nano-0.8-int8",
    )

    assert len(entries) == 8
    assert {entry["voice_alias"] for entry in entries} == set(aliases)
    assert {entry["internal_voice"] for entry in entries} == set(DEFAULT_VOICE_ALIASES.values())
    assert len({entry["calibration_key"] for entry in entries}) == 8
    assert all(entry["calibration_key"].startswith("kitten:nano-0.8-int8:") for entry in entries)


def test_identity_expansion_accepts_internal_voice_filter():
    entries = expand_identities(
        tuple(DEFAULT_VOICE_ALIASES),
        FakeVoiceBank(),
        requested_model="nano-0.8-int8",
        model_id="nano-0.8-int8",
        model_ref="kitten:nano-0.8-int8",
        selected_voices=["expr-voice-2-m"],
    )

    assert len(entries) == 1
    assert entries[0]["voice_alias"] == "Jasper"


def test_measurement_failures_record_identity_phase_and_error(monkeypatch):
    class FailingModel(FakeModel):
        def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
            raise RuntimeError("synthetic inference failure")

    fake_model = FailingModel()
    entries = expand_identities(
        ("Jasper",),
        fake_model.voice_bank,
        requested_model="nano-0.8-int8",
        model_id="nano-0.8-int8",
        model_ref="kitten:nano-0.8-int8",
    )
    monkeypatch.setattr(
        "benchmarks.voice_level_benchmark.measure_loudness",
        lambda audio, *, sample_rate: pytest.fail("failed synthesis must not be measured"),
    )

    measurements, failures = measure_identities(fake_model, entries, stimuli()[:1], load_policy())

    assert measurements == []
    assert len(failures) == 3
    assert all(row["phase"] == "synthesis" for row in failures)
    assert all(row["error_type"] == "RuntimeError" for row in failures)
    assert all(row["calibration_key"] == entries[0]["calibration_key"] for row in failures)
    assert all(row["stimulus_id"] == "short" for row in failures)


def test_measurement_loop_reuses_model_and_disables_calibration(monkeypatch):
    fake_model = FakeModel()
    entries = expand_identities(
        ("Bella", "Jasper"),
        fake_model.voice_bank,
        requested_model="nano-0.8-int8",
        model_id="nano-0.8-int8",
        model_ref="kitten:nano-0.8-int8",
    )
    monkeypatch.setattr(
        "benchmarks.voice_level_benchmark.measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-22.0),
    )

    measurements, failures = measure_identities(fake_model, entries, stimuli(), load_policy())

    assert len(fake_model.calls) == 2 * 3 * 3
    assert all(call["config"].voice_level.mode == "off" for call in fake_model.calls)
    assert len(measurements) == 18
    assert failures == []
    assert all(row["integrated_lufs"] == -22.0 for row in measurements)
    assert all(row["raw_peak"] == pytest.approx(0.2) for row in measurements)


def test_per_stimulus_repeats_aggregate_to_median_of_stimulus_medians():
    entry = identity("Jasper", "expr-voice-2-m")
    measurements = []
    for stimulus_id, center in (("short", -20.0), ("medium", -22.0), ("long", -24.0)):
        for repeat, value in enumerate((center - 0.1, center, center + 0.1)):
            measurements.append(
                {
                    **entry,
                    "stimulus_id": stimulus_id,
                    "repeat": repeat,
                    "integrated_lufs": value,
                    "raw_peak": 0.2,
                }
            )

    aggregate = aggregate_measurements([entry], measurements, stimuli(), load_policy())[0]

    assert aggregate["stimulus_medians"] == {"short": -20.0, "medium": -22.0, "long": -24.0}
    assert aggregate["median_lufs"] == -22.0
    assert aggregate["stimulus_spread_lu"] == 4.0
    assert aggregate["mad_lu"] == pytest.approx(0.1)
    assert aggregate["gain_db"] == -2.0
    assert aggregate["status"] == "eligible"


def test_repeat_mad_threshold_marks_high_variability():
    entry = identity("Jasper", "expr-voice-2-m")
    measurements = []
    for stimulus in stimuli():
        for repeat, value in enumerate((-20.0, -18.0, -16.0)):
            measurements.append(
                {
                    **entry,
                    "stimulus_id": stimulus["id"],
                    "repeat": repeat,
                    "integrated_lufs": value,
                    "raw_peak": 0.2,
                }
            )

    aggregate = aggregate_measurements([entry], measurements, stimuli(), load_policy())[0]

    assert aggregate["mad_lu"] == 2.0
    assert aggregate["status"] == "high_variability"


def test_missing_stimulus_repeat_marks_identity_incomplete():
    entry = identity("Jasper", "expr-voice-2-m")
    measurements = synthetic_measurements(entry, values_around(-22.0))
    measurements = [
        row for row in measurements if not (row["stimulus_id"] == "medium" and row["repeat"] == 2)
    ]

    aggregate = aggregate_measurements([entry], measurements, stimuli(), load_policy())[0]
    report = report_for([entry], measurements)

    assert aggregate["status"] == "incomplete"
    assert aggregate["complete"] is False
    assert report["coverage"] == {
        "identities_expected": 1,
        "identities_measured": 0,
        "identities_failed": 1,
        "measurement_failures": 0,
        "complete": False,
    }


def test_full_coverage_report_counts_identities_and_samples():
    entries = [
        identity("Bella", "expr-voice-2-f"),
        identity("Jasper", "expr-voice-2-m"),
    ]
    measurements = []
    for entry in entries:
        measurements.extend(synthetic_measurements(entry, values_around(-22.0)))

    report = report_for(entries, measurements)

    assert report["coverage"] == {
        "identities_expected": 2,
        "identities_measured": 2,
        "identities_failed": 0,
        "measurement_failures": 0,
        "complete": True,
    }
    assert len(report["measurements"]) == 18
    assert all("predicted_peak_at_gain_db" in row for row in report["measurements"])


def test_partial_promotion_fails_without_explicit_allowance():
    entry = identity("Jasper", "expr-voice-2-m")
    measurements = synthetic_measurements(entry, values_around(-22.0))[:-1]
    report = report_for([entry], measurements)

    with pytest.raises(ValueError, match="--allow-partial"):
        build_candidate(report)


def test_high_variability_is_excluded_unless_explicitly_allowed():
    variable = identity("Bella", "expr-voice-2-f")
    stable = identity("Jasper", "expr-voice-2-m")
    measurements = []
    for stimulus in stimuli():
        for repeat, value in enumerate((-20.0, -18.0, -16.0)):
            measurements.append(
                {
                    **variable,
                    "stimulus_id": stimulus["id"],
                    "repeat": repeat,
                    "requested_speed": 1.0,
                    "effective_speed": 1.0,
                    "sample_rate": 24000,
                    "duration_seconds": 1.0,
                    "calibration_mode": "off",
                    "integrated_lufs": value,
                    "raw_peak": 0.2,
                }
            )
    measurements.extend(synthetic_measurements(stable, values_around(-22.0)))
    report = report_for([variable, stable], measurements)

    candidate, counts = build_candidate(report)

    assert set(candidate["voices"]) == {stable["calibration_key"]}
    assert counts["high_variability"] == 1
    assert counts["eligible"] == 1


def test_candidate_catalog_loads_with_runtime_loader(tmp_path):
    entry = identity("Jasper", "expr-voice-2-m")
    report = report_for([entry], synthetic_measurements(entry, values_around(-22.0)))
    candidate, _ = build_candidate(report)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")

    loaded = load_voice_calibration(path)

    assert list(loaded.voices) == [VoiceCalibrationKey.parse(entry["calibration_key"])]
    record = loaded.voices[VoiceCalibrationKey.parse(entry["calibration_key"])]
    assert record.samples == 9
    assert record.gain_db == -2.0


def test_gain_calculation_honors_both_policy_bounds():
    loud = identity("Bella", "expr-voice-2-f")
    quiet = identity("Jasper", "expr-voice-2-m")
    measurements = []
    measurements.extend(synthetic_measurements(loud, values_around(-40.0)))
    measurements.extend(synthetic_measurements(quiet, values_around(-10.0)))

    aggregates = aggregate_measurements([loud, quiet], measurements, stimuli(), load_policy())

    by_key = {row["calibration_key"]: row for row in aggregates}
    assert by_key[loud["calibration_key"]]["gain_db"] == 8.0
    assert by_key[loud["calibration_key"]]["gain_limited"] is True
    assert by_key[quiet["calibration_key"]]["gain_db"] == -12.0
    assert by_key[quiet["calibration_key"]]["gain_limited"] is True


def test_positive_gain_peak_risk_requires_explicit_review():
    entry = identity("Bella", "expr-voice-2-f")
    report = report_for(
        [entry],
        synthetic_measurements(entry, values_around(-30.0), raw_peak=0.6),
    )

    assert report["aggregates"][0]["peak_safety_review_required"] is True
    with pytest.raises(ValueError, match="explicit peak-risk review"):
        build_candidate(report)
    candidate, _ = build_candidate(report, allow_peak_risk=True)
    assert entry["calibration_key"] in candidate["voices"]


def test_gain_limited_candidate_requires_explicit_review():
    entry = identity("Bella", "expr-voice-2-f")
    report = report_for([entry], synthetic_measurements(entry, values_around(-40.0)))

    with pytest.raises(ValueError, match="explicit review is required"):
        build_candidate(report)
    candidate, _ = build_candidate(report, allow_gain_limited=True)
    assert candidate["voices"][entry["calibration_key"]]["gain_db"] == 8.0


def test_candidate_rejects_non_finite_measurement_report():
    entry = identity("Jasper", "expr-voice-2-m")
    report = report_for([entry], synthetic_measurements(entry, values_around(-22.0)))
    report["measurements"][0]["integrated_lufs"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        build_candidate(report)


def test_promoter_refuses_production_catalog_path(tmp_path):
    with pytest.raises(SystemExit, match="never writes directly"):
        main([str(tmp_path / "does-not-exist.json"), "--output", str(PRODUCTION_CATALOG)])
