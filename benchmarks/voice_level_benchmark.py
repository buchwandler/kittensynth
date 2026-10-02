#!/usr/bin/env python3
"""Measure prepared Kitten speech for static per-model, per-voice calibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from audiosig import measure_loudness

from kittensynth import KittenVoice, SynthesisConfig, VoiceLevelConfig
from kittensynth.voice_level import VoiceCalibrationKey

POLICY_PATH = Path(__file__).with_name("data") / "voice_level_policy.json"
STIMULI_PATH = Path(__file__).with_name("data") / "voice_level_stimuli.json"
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")
_POLICY_FIELDS = {
    "schema",
    "name",
    "repeats",
    "reference_lufs",
    "min_gain_db",
    "max_gain_db",
    "max_mad_lu",
    "identity_overrides",
}
_STIMULI_FIELDS = {"schema", "name", "language", "stimuli"}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load and validate the measurement and gain-bound policy."""
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, Mapping):
        raise ValueError("voice-level policy must be an object")
    unknown = set(policy) - _POLICY_FIELDS
    if unknown:
        raise ValueError(f"voice-level policy has unknown fields: {sorted(unknown)}")
    schema = policy.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError("voice-level policy has an unsupported schema")
    if not isinstance(policy.get("name"), str) or not policy["name"].strip():
        raise ValueError("voice-level policy name must be non-empty")
    repeats = policy.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("policy repeats must be a positive integer")
    for name in ("reference_lufs", "min_gain_db", "max_gain_db", "max_mad_lu"):
        _finite(policy.get(name), f"policy {name}")
    if policy["min_gain_db"] > policy["max_gain_db"] or policy["max_mad_lu"] < 0:
        raise ValueError("voice-level policy limits are invalid")
    identity_overrides = policy.get("identity_overrides", {})
    if not isinstance(identity_overrides, Mapping):
        raise ValueError("policy identity_overrides must be an object")
    for calibration_key, override in identity_overrides.items():
        if not isinstance(calibration_key, str) or not isinstance(override, Mapping):
            raise ValueError("policy identity overrides must map calibration keys to objects")
        if set(override) - {"min_gain_db", "rationale"}:
            raise ValueError(f"policy identity override for {calibration_key} has unknown fields")
        VoiceCalibrationKey.parse(calibration_key)
        minimum_gain_db = _finite(
            override.get("min_gain_db"), f"policy {calibration_key}.min_gain_db"
        )
        if minimum_gain_db > policy["max_gain_db"]:
            raise ValueError(f"policy identity override for {calibration_key} exceeds max_gain_db")
        rationale = override.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"policy identity override for {calibration_key} needs a rationale")
    return dict(policy)


def load_stimuli(path: Path = STIMULI_PATH) -> dict[str, Any]:
    """Load prepared speakable text stimuli with unique IDs."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or set(document) != _STIMULI_FIELDS:
        raise ValueError("stimuli document has an invalid shape")
    schema = document.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError("stimuli document has an unsupported schema")
    if not isinstance(document.get("name"), str) or not document["name"].strip():
        raise ValueError("stimuli name must be non-empty")
    language = document.get("language")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("stimuli language must be non-empty")
    stimuli = document.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise ValueError("stimuli must be a non-empty list")
    seen: set[str] = set()
    validated: list[dict[str, str]] = []
    for stimulus in stimuli:
        if not isinstance(stimulus, Mapping) or set(stimulus) != {"id", "text"}:
            raise ValueError("each stimulus must contain only id and text")
        stimulus_id = stimulus.get("id")
        text = stimulus.get("text")
        if not isinstance(stimulus_id, str) or not stimulus_id.strip() or stimulus_id in seen:
            raise ValueError("stimulus IDs must be unique non-empty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"stimulus {stimulus_id} must contain prepared speakable text")
        seen.add(stimulus_id)
        validated.append({"id": stimulus_id, "text": text})
    return {"schema": 1, "name": document["name"], "language": language, "stimuli": validated}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def generated_with() -> dict[str, str]:
    return {
        name: _package_version(name)
        for name in ("kittensynth", "kitteng2p", "onnxvoice", "audiosig")
    }


def expand_identities(
    available_voices: Sequence[str],
    voice_bank: Any,
    *,
    requested_model: str,
    model_id: str,
    model_ref: str,
    selected_voices: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve every available alias to a unique managed/internal identity."""
    selected = set(selected_voices or ())
    entries: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for alias in sorted(available_voices, key=str.casefold):
        internal_voice = voice_bank.resolve(alias)
        if selected and alias not in selected and internal_voice not in selected:
            continue
        key = str(VoiceCalibrationKey("kitten", model_id, internal_voice))
        if key in seen_keys:
            raise ValueError(f"multiple available aliases resolve to calibration identity {key}")
        seen_keys.add(key)
        entries.append(
            {
                "model_source": "kitten",
                "requested_model": requested_model,
                "model_id": model_id,
                "model_ref": model_ref,
                "voice_alias": alias,
                "internal_voice": internal_voice,
                "calibration_key": key,
            }
        )
    if selected and not entries:
        raise ValueError(f"no available voice matches: {', '.join(sorted(selected))}")
    return entries


def _failure(
    entry: Mapping[str, Any],
    stimulus_id: str,
    repeat: int,
    error: Exception,
    *,
    phase: str,
) -> dict[str, Any]:
    return {
        **dict(entry),
        "stimulus_id": stimulus_id,
        "repeat": repeat,
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
    }


class _SynthesisModel(Protocol):
    def synthesize_prepared(self, text: str, *, voice: str, config: SynthesisConfig) -> Any: ...


def measure_identities(
    model: _SynthesisModel,
    entries: Sequence[Mapping[str, Any]],
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure all requested identities/stimuli/repeats on one opened model."""
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repeats = int(policy["repeats"])
    config = SynthesisConfig(speed=1.0, voice_level=VoiceLevelConfig(mode="off"))
    total = len(entries) * len(stimuli) * repeats
    completed = 0
    for entry in entries:
        for stimulus in stimuli:
            for repeat in range(repeats):
                completed += 1
                print(
                    f"[{completed}/{total}] {entry['calibration_key']} "
                    f"{stimulus['id']} repeat {repeat + 1}",
                    flush=True,
                )
                try:
                    result = model.synthesize_prepared(
                        stimulus["text"],
                        voice=str(entry["voice_alias"]),
                        config=config,
                    )
                except Exception as error:
                    failures.append(
                        _failure(entry, stimulus["id"], repeat, error, phase="synthesis")
                    )
                    continue
                try:
                    audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    if not audio.size:
                        raise ValueError("synthesis audio is empty")
                    if not np.all(np.isfinite(audio)):
                        raise ValueError("synthesis audio contains non-finite samples")
                    sample_rate = int(result.sample_rate)
                    if sample_rate <= 0:
                        raise ValueError("sample rate must be positive")
                    peak = float(np.max(np.abs(audio)))
                    loudness = measure_loudness(audio, sample_rate=sample_rate)
                    measured_lufs = loudness.integrated_lufs
                    if measured_lufs is None:
                        raise ValueError("integrated loudness is unavailable")
                    integrated_lufs = float(measured_lufs)
                    if not math.isfinite(integrated_lufs):
                        raise ValueError("integrated loudness is not finite")
                    duration = float(result.duration_seconds)
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("duration must be finite and positive")
                    effective_speed = float(result.speed)
                    if not math.isfinite(effective_speed) or effective_speed <= 0:
                        raise ValueError("effective speed must be finite and positive")
                except Exception as error:
                    failures.append(
                        _failure(entry, stimulus["id"], repeat, error, phase="measurement")
                    )
                    continue
                measurements.append(
                    {
                        **dict(entry),
                        "stimulus_id": stimulus["id"],
                        "repeat": repeat,
                        "requested_speed": 1.0,
                        "effective_speed": effective_speed,
                        "sample_rate": sample_rate,
                        "duration_seconds": duration,
                        "integrated_lufs": integrated_lufs,
                        "raw_peak": peak,
                        "calibration_mode": "off",
                    }
                )
    return measurements, failures


def _gain_for(
    median_lufs: float, calibration_key: str, policy: Mapping[str, Any]
) -> tuple[float, bool]:
    identity_override = policy.get("identity_overrides", {}).get(calibration_key)
    minimum_gain = (
        float(identity_override["min_gain_db"])
        if identity_override is not None
        else float(policy["min_gain_db"])
    )
    maximum_gain = float(policy["max_gain_db"])
    requested_gain = float(policy["reference_lufs"]) - median_lufs
    gain = min(maximum_gain, max(minimum_gain, requested_gain))
    limited = math.isclose(gain, minimum_gain, abs_tol=1e-12) or math.isclose(
        gain, maximum_gain, abs_tol=1e-12
    )
    return gain, limited


def aggregate_measurements(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate repeats within each stimulus, then median stimulus loudness."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        groups[str(measurement["calibration_key"])].append(measurement)
    repeats = int(policy["repeats"])
    expected_repeats = set(range(repeats))
    stimulus_ids = [str(stimulus["id"]) for stimulus in stimuli]
    aggregates: list[dict[str, Any]] = []

    for entry in entries:
        calibration_key = str(entry["calibration_key"])
        identity_rows = groups.get(calibration_key, [])
        by_stimulus: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in identity_rows:
            by_stimulus[str(row["stimulus_id"])].append(row)
        stimulus_medians: dict[str, float] = {}
        repeat_mads: dict[str, float] = {}
        complete = True
        for stimulus_id in stimulus_ids:
            rows = by_stimulus.get(stimulus_id, [])
            repeat_values: dict[int, float] = {}
            duplicate_repeat = False
            for row in rows:
                repeat = row.get("repeat")
                if isinstance(repeat, bool) or not isinstance(repeat, int):
                    duplicate_repeat = True
                    continue
                if repeat in repeat_values:
                    duplicate_repeat = True
                repeat_values[repeat] = _finite(row.get("integrated_lufs"), "integrated_lufs")
            if set(repeat_values) != expected_repeats or duplicate_repeat:
                complete = False
                continue
            values = list(repeat_values.values())
            median = statistics.median(values)
            stimulus_medians[stimulus_id] = median
            repeat_mads[stimulus_id] = statistics.median(abs(value - median) for value in values)
        if set(stimulus_medians) != set(stimulus_ids):
            complete = False

        median_lufs = statistics.median(stimulus_medians.values()) if stimulus_medians else None
        mad_lu = max(repeat_mads.values(), default=0.0)
        stimulus_spread = (
            max(stimulus_medians.values()) - min(stimulus_medians.values())
            if stimulus_medians
            else 0.0
        )
        status = "incomplete" if not complete else "eligible"
        if complete and mad_lu > float(policy["max_mad_lu"]):
            status = "high_variability"
        gain_db: float | None = None
        gain_limited = False
        if median_lufs is not None:
            gain_db, gain_limited = _gain_for(median_lufs, calibration_key, policy)

        peak_scale = 10.0 ** ((gain_db or 0.0) / 20.0)
        predicted_peaks = [
            _finite(row.get("raw_peak"), "raw_peak") * peak_scale for row in identity_rows
        ]
        peak_safety_review_required = (
            gain_db is not None and gain_db > 0 and any(peak > 1.0 for peak in predicted_peaks)
        )
        aggregates.append(
            {
                **dict(entry),
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "repeat_mad_by_stimulus": repeat_mads,
                "stimulus_medians": stimulus_medians,
                "stimulus_spread_lu": stimulus_spread,
                "repeat_count": len(identity_rows),
                "stimulus_count": len(stimulus_medians),
                "status": status,
                "gain_db": gain_db,
                "gain_limited": gain_limited,
                "max_raw_peak": max(
                    (_finite(row.get("raw_peak"), "raw_peak") for row in identity_rows), default=0.0
                ),
                "predicted_peak": max(predicted_peaks, default=0.0),
                "peak_safety_review_required": peak_safety_review_required,
                "complete": complete,
            }
        )
    return aggregates


def build_report(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
    language: str = "en-us",
) -> dict[str, Any]:
    aggregates = aggregate_measurements(entries, measurements, stimuli, policy)
    measured = sum(1 for row in aggregates if row["complete"])
    expected = len(entries)
    gains = {str(row["calibration_key"]): row["gain_db"] for row in aggregates}
    detailed_measurements = []
    for row in measurements:
        gain = gains.get(str(row["calibration_key"]))
        predicted_peak = (
            float(row["raw_peak"]) * (10.0 ** (float(gain) / 20.0)) if gain is not None else None
        )
        detailed_measurements.append({**dict(row), "predicted_peak_at_gain_db": predicted_peak})
    return {
        "schema": 1,
        "corpus": str(policy["name"]),
        "language": language,
        "generated_with": generated_with(),
        "policy": dict(policy),
        "coverage": {
            "identities_expected": expected,
            "identities_measured": measured,
            "identities_failed": expected - measured,
            "measurement_failures": len(failures),
            "complete": expected == measured and not failures,
        },
        "stimuli": [dict(stimulus) for stimulus in stimuli],
        "measurements": detailed_measurements,
        "failures": [dict(row) for row in failures],
        "aggregates": aggregates,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write JSON atomically so an interrupted run cannot leave a partial report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _voice_open_failures(
    requested_model: str,
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
    error: Exception,
) -> list[dict[str, Any]]:
    failures = []
    for stimulus in stimuli:
        for repeat in range(int(policy["repeats"])):
            failures.append(
                {
                    "model_source": "kitten",
                    "requested_model": requested_model,
                    "model_id": None,
                    "model_ref": None,
                    "voice_alias": None,
                    "internal_voice": None,
                    "calibration_key": None,
                    "stimulus_id": stimulus["id"],
                    "repeat": repeat,
                    "phase": "voice_open",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nano-0.8-int8")
    parser.add_argument("--quality")
    parser.add_argument("--voice", action="append", help="limit to a voice alias or internal ID")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli", type=Path, default=STIMULI_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.policy)
    stimulus_document = load_stimuli(args.stimuli)
    stimuli = stimulus_document["stimuli"]
    selected_voices = args.voice or []
    try:
        model = KittenVoice.from_pretrained(
            args.model,
            quality=args.quality,
            cache_dir=args.cache_dir,
            offline=args.offline,
            refresh_catalog=args.refresh_catalog,
        )
    except Exception as error:
        if args.list_only:
            print(f"Unable to open managed Kitten model {args.model!r}: {error}")
            return 1
        failures = _voice_open_failures(args.model, stimuli, policy, error)
        report = build_report(
            [],
            [],
            failures,
            stimuli=stimuli,
            policy=policy,
            language=stimulus_document["language"],
        )
        write_report(args.output, report)
        print(f"Model open failed; failure report: {args.output}")
        return 1

    with model:
        if not model.model_id:
            print("Opened Kitten model has no stable managed model ID.")
            return 1
        entries = expand_identities(
            model.available_voices,
            model.voice_bank,
            requested_model=args.model,
            model_id=model.model_id,
            model_ref=model.model_ref or "",
            selected_voices=selected_voices,
        )
        if args.list_only:
            for entry in entries:
                print(
                    f"{entry['voice_alias']}\t{entry['internal_voice']}\t{entry['calibration_key']}"
                )
            print(f"Identities: {len(entries)}")
            return 0
        measurements, failures = measure_identities(model, entries, stimuli, policy)
    report = build_report(
        entries,
        measurements,
        failures,
        stimuli=stimuli,
        policy=policy,
        language=stimulus_document["language"],
    )
    write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Report: {args.output}")
    print(
        "Coverage: "
        f"{coverage['identities_measured']}/{coverage['identities_expected']} identities; "
        f"{coverage['measurement_failures']} measurement failure(s)"
    )
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
