#!/usr/bin/env python3
"""Build a reviewable candidate catalog from real Kitten calibration measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kittensynth.voice_level import VoiceCalibrationKey

PRODUCTION_CATALOG = (
    Path(__file__).resolve().parents[1] / "kittensynth" / "data" / "voice_level_calibration.json"
)
DEFAULT_INPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/candidate_catalog.json")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _reject_non_finite(value: Any, location: str = "report") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _finite_number(value, location)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{location}[{index}]")


def _validated_policy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("measurement report policy must be an object")
    if set(raw) - {
        "schema",
        "name",
        "repeats",
        "reference_lufs",
        "min_gain_db",
        "max_gain_db",
        "max_mad_lu",
        "identity_overrides",
    }:
        raise ValueError("measurement policy has unknown fields")
    if (
        not isinstance(raw.get("schema"), int)
        or isinstance(raw.get("schema"), bool)
        or raw.get("schema") != 1
    ):
        raise ValueError("unsupported measurement policy schema")
    _required_string(raw.get("name"), "policy.name")
    repeats = _integer(raw.get("repeats"), "policy.repeats", minimum=1)
    reference = _finite_number(raw.get("reference_lufs"), "policy.reference_lufs")
    minimum = _finite_number(raw.get("min_gain_db"), "policy.min_gain_db")
    maximum = _finite_number(raw.get("max_gain_db"), "policy.max_gain_db")
    max_mad = _finite_number(raw.get("max_mad_lu"), "policy.max_mad_lu")
    if minimum > maximum or max_mad < 0:
        raise ValueError("measurement policy limits are invalid")
    overrides = raw.get("identity_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("measurement policy identity_overrides must be an object")
    for key_string, override in overrides.items():
        if not isinstance(key_string, str) or not isinstance(override, Mapping):
            raise ValueError("measurement identity overrides must map keys to objects")
        if set(override) - {"min_gain_db", "rationale"}:
            raise ValueError(f"{key_string} identity override has unknown fields")
        VoiceCalibrationKey.parse(key_string)
        override_minimum = _finite_number(override.get("min_gain_db"), f"{key_string}.min_gain_db")
        if override_minimum > maximum:
            raise ValueError(f"{key_string}.min_gain_db exceeds max_gain_db")
        _required_string(override.get("rationale"), f"{key_string}.rationale")
    return {
        **dict(raw),
        "repeats": repeats,
        "reference_lufs": reference,
        "min_gain_db": minimum,
        "max_gain_db": maximum,
        "max_mad_lu": max_mad,
        "identity_overrides": dict(overrides),
    }


def _validate_report_shape(
    report: Mapping[str, Any], *, allow_partial: bool
) -> tuple[dict[str, Any], Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    schema = report.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != 1:
        raise ValueError("unsupported measurement report schema")
    _reject_non_finite(report)
    policy = _validated_policy(report.get("policy"))
    corpus = _required_string(report.get("corpus"), "report.corpus")
    if corpus != policy["name"]:
        raise ValueError("report corpus does not match the declared policy")
    stimuli = report.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise ValueError("measurement report must contain prepared stimuli")
    stimulus_ids: set[str] = set()
    for stimulus in stimuli:
        if not isinstance(stimulus, Mapping):
            raise ValueError("stimulus entries must be objects")
        stimulus_id = _required_string(stimulus.get("id"), "stimulus.id")
        _required_string(stimulus.get("text"), f"stimulus {stimulus_id}.text")
        if stimulus_id in stimulus_ids:
            raise ValueError(f"duplicate stimulus ID: {stimulus_id}")
        stimulus_ids.add(stimulus_id)
    generated_with = report.get("generated_with")
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in generated_with.items()
    ):
        raise ValueError("report generated_with must be a string mapping")
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("measurement report is missing coverage")
    expected = _integer(coverage.get("identities_expected"), "identities_expected")
    measured = _integer(coverage.get("identities_measured"), "identities_measured")
    failed = _integer(coverage.get("identities_failed"), "identities_failed")
    measurement_failures = _integer(coverage.get("measurement_failures"), "measurement_failures")
    if measured > expected or failed != expected - measured:
        raise ValueError("measurement coverage counts are inconsistent")
    failures = report.get("failures")
    aggregates = report.get("aggregates")
    measurements = report.get("measurements")
    if (
        not isinstance(failures, list)
        or not isinstance(aggregates, list)
        or not isinstance(measurements, list)
    ):
        raise ValueError("measurement report failures, measurements, and aggregates must be lists")
    if any(not isinstance(item, Mapping) for item in failures):
        raise ValueError("failure entries must be objects")
    if len(failures) != measurement_failures:
        raise ValueError("measurement failure count does not match failure records")
    if len(aggregates) != expected:
        raise ValueError("aggregate count does not match expected identity coverage")

    policy_repeats = int(policy["repeats"])
    aggregate_by_key: dict[str, Mapping[str, Any]] = {}
    for aggregate in aggregates:
        if not isinstance(aggregate, Mapping):
            raise ValueError("aggregate entries must be objects")
        model_source = _required_string(aggregate.get("model_source"), "model_source")
        model_id = _required_string(aggregate.get("model_id"), "model_id")
        internal_voice = _required_string(aggregate.get("internal_voice"), "internal_voice")
        key = VoiceCalibrationKey(model_source, model_id, internal_voice)
        if aggregate.get("calibration_key") != str(key) or str(key) in aggregate_by_key:
            raise ValueError(f"invalid or duplicate calibration key: {key}")
        aggregate_by_key[str(key)] = aggregate

    measurements_by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unique_samples: set[tuple[str, str, int]] = set()
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError("measurement entries must be objects")
        key = _required_string(measurement.get("calibration_key"), "measurement.calibration_key")
        aggregate = aggregate_by_key.get(key)
        if aggregate is None:
            raise ValueError(f"measurement references an unknown calibration identity: {key}")
        stimulus_id = _required_string(measurement.get("stimulus_id"), "measurement.stimulus_id")
        if stimulus_id not in stimulus_ids:
            raise ValueError(f"measurement references an unknown stimulus: {stimulus_id}")
        repeat = _integer(measurement.get("repeat"), f"{key}.{stimulus_id}.repeat")
        if repeat >= policy_repeats:
            raise ValueError(f"{key}.{stimulus_id} repeat is outside the declared policy")
        sample_key = (key, stimulus_id, repeat)
        if sample_key in unique_samples:
            raise ValueError(f"duplicate measurement sample: {key}/{stimulus_id}/{repeat}")
        unique_samples.add(sample_key)
        for identity_field in ("model_source", "model_id", "voice_alias", "internal_voice"):
            if measurement.get(identity_field) != aggregate.get(identity_field):
                raise ValueError(f"measurement {key} has inconsistent {identity_field}")
        if measurement.get("calibration_mode") != "off":
            raise ValueError(f"measurement {key} was not synthesized with calibration disabled")
        _integer(measurement.get("sample_rate"), f"{key}.sample_rate", minimum=1)
        duration = _finite_number(measurement.get("duration_seconds"), f"{key}.duration_seconds")
        requested_speed = _finite_number(
            measurement.get("requested_speed"), f"{key}.requested_speed"
        )
        effective_speed = _finite_number(
            measurement.get("effective_speed"), f"{key}.effective_speed"
        )
        _finite_number(measurement.get("integrated_lufs"), f"{key}.integrated_lufs")
        raw_peak = _finite_number(measurement.get("raw_peak"), f"{key}.raw_peak")
        if duration <= 0 or requested_speed <= 0 or effective_speed <= 0 or raw_peak < 0:
            raise ValueError(f"measurement {key} contains an invalid duration, speed, or peak")
        predicted = measurement.get("predicted_peak_at_gain_db")
        if predicted is not None:
            _finite_number(predicted, f"{key}.predicted_peak_at_gain_db")
        measurements_by_key[key].append(measurement)

    for key, aggregate in aggregate_by_key.items():
        rows = measurements_by_key.get(key, [])
        actual_samples = {(str(row["stimulus_id"]), int(row["repeat"])) for row in rows}
        expected_samples = {
            (stimulus_id, repeat)
            for stimulus_id in stimulus_ids
            for repeat in range(policy_repeats)
        }
        is_complete = actual_samples == expected_samples
        if aggregate.get("complete") is not is_complete:
            raise ValueError(f"{key} completeness does not match its measurement samples")
        gain = aggregate.get("gain_db")
        for row in rows:
            predicted = row.get("predicted_peak_at_gain_db")
            expected_peak = (
                float(row["raw_peak"]) * (10.0 ** (float(gain) / 20.0))
                if gain is not None
                else None
            )
            if expected_peak is None:
                if predicted is not None:
                    raise ValueError(
                        f"{key} incomplete measurement has an unexpected predicted peak"
                    )
            elif predicted is None or not math.isclose(
                _finite_number(predicted, f"{key}.predicted_peak_at_gain_db"),
                expected_peak,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{key} measured predicted peak does not match its gain")

    completed = sum(1 for aggregate in aggregates if aggregate.get("complete") is True)
    if completed != measured:
        raise ValueError("measured identity count does not match complete aggregates")
    is_complete = expected == measured and failed == 0 and measurement_failures == 0
    if coverage.get("complete") is not is_complete:
        raise ValueError("measurement coverage completeness flag is inconsistent")
    if not is_complete and not allow_partial:
        raise ValueError("incomplete identity coverage requires --allow-partial")
    return policy, coverage, aggregates, failures


def _verify_aggregate_from_measurements(
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    stimulus_ids: Sequence[str],
    repeats: int,
) -> None:
    key = str(aggregate["calibration_key"])
    by_stimulus: dict[str, dict[int, float]] = {}
    raw_peaks: list[float] = []
    for row in rows:
        stimulus_id = str(row["stimulus_id"])
        repeat = int(row["repeat"])
        by_stimulus.setdefault(stimulus_id, {})[repeat] = _finite_number(
            row.get("integrated_lufs"), f"{key}.integrated_lufs"
        )
        raw_peaks.append(_finite_number(row.get("raw_peak"), f"{key}.raw_peak"))
    medians: dict[str, float] = {}
    repeat_mads: dict[str, float] = {}
    expected_repeats = set(range(repeats))
    for stimulus_id in stimulus_ids:
        values_by_repeat = by_stimulus.get(stimulus_id, {})
        if set(values_by_repeat) != expected_repeats:
            raise ValueError(f"{key} does not contain all repeats for {stimulus_id}")
        values = list(values_by_repeat.values())
        median = statistics.median(values)
        medians[stimulus_id] = median
        repeat_mads[stimulus_id] = statistics.median(abs(value - median) for value in values)
    identity_median = statistics.median(medians.values())
    identity_mad = max(repeat_mads.values(), default=0.0)
    spread = max(medians.values()) - min(medians.values())
    raw_peak = max(raw_peaks, default=0.0)
    gain = _finite_number(aggregate.get("gain_db"), f"{key}.gain_db")
    predicted_peaks = [peak * (10.0 ** (gain / 20.0)) for peak in raw_peaks]
    predicted_peak = max(predicted_peaks, default=0.0)
    expected_values = {
        "median_lufs": identity_median,
        "mad_lu": identity_mad,
        "stimulus_spread_lu": spread,
        "max_raw_peak": raw_peak,
        "predicted_peak": predicted_peak,
    }
    for name, expected in expected_values.items():
        actual = _finite_number(aggregate.get(name), f"{key}.{name}")
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{key}.{name} does not match raw measurements")
    for field, expected_map in (
        ("stimulus_medians", medians),
        ("repeat_mad_by_stimulus", repeat_mads),
    ):
        actual_map = aggregate.get(field)
        if not isinstance(actual_map, Mapping) or set(actual_map) != set(expected_map):
            raise ValueError(f"{key}.{field} does not match raw measurements")
        for stimulus_id, expected in expected_map.items():
            actual = _finite_number(actual_map[stimulus_id], f"{key}.{field}.{stimulus_id}")
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{key}.{field} does not match raw measurements")
    peak_review = gain > 0 and any(peak > 1.0 for peak in predicted_peaks)
    if aggregate.get("peak_safety_review_required") is not peak_review:
        raise ValueError(f"{key} peak safety flag does not match raw measurements")


def build_candidate(
    report: Mapping[str, Any],
    *,
    allow_partial: bool = False,
    include_high_variability: bool = False,
    allow_peak_risk: bool = False,
    allow_gain_limited: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Validate measured coverage and derive a separate candidate catalog."""
    policy, _, aggregates, _ = _validate_report_shape(report, allow_partial=allow_partial)
    generated_with = report["generated_with"]
    corpus = str(report["corpus"])
    reference_lufs = float(policy["reference_lufs"])
    repeats = int(policy["repeats"])
    stimulus_count = len(report["stimuli"])
    measurements_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for measurement in report["measurements"]:
        measurements_by_key.setdefault(str(measurement["calibration_key"]), []).append(measurement)
    stimulus_ids = [str(stimulus["id"]) for stimulus in report["stimuli"]]
    seen_keys: set[str] = set()
    identity_overrides = policy["identity_overrides"]
    voices: dict[str, dict[str, Any]] = {}
    counts = {"eligible": 0, "high_variability": 0, "incomplete": 0}

    for aggregate in aggregates:
        if not isinstance(aggregate, Mapping):
            raise ValueError("aggregate entries must be objects")
        model_source = _required_string(aggregate.get("model_source"), "model_source")
        model_id = _required_string(aggregate.get("model_id"), "model_id")
        internal_voice = _required_string(aggregate.get("internal_voice"), "internal_voice")
        voice_alias = _required_string(aggregate.get("voice_alias"), "voice_alias")
        if voice_alias == internal_voice:
            raise ValueError("calibration identity must use the internal voice ID, not an alias")
        key = VoiceCalibrationKey(model_source, model_id, internal_voice)
        if aggregate.get("calibration_key") != str(key) or str(key) in seen_keys:
            raise ValueError(f"invalid or duplicate calibration key: {key}")
        seen_keys.add(str(key))
        complete = aggregate.get("complete")
        status = aggregate.get("status")
        if not isinstance(complete, bool):
            raise ValueError(f"{key}.complete must be a boolean")
        if status not in {"eligible", "high_variability", "incomplete"}:
            raise ValueError(f"{key} has unknown status {status!r}")
        repeat_count = _integer(aggregate.get("repeat_count"), f"{key}.repeat_count")
        measured_stimuli = _integer(aggregate.get("stimulus_count"), f"{key}.stimulus_count")
        if complete != (
            measured_stimuli == stimulus_count and repeat_count == stimulus_count * repeats
        ):
            raise ValueError(f"{key} completeness does not match stimulus/repeat counts")
        if not complete:
            if status != "incomplete":
                raise ValueError(f"{key} incomplete measurements have status {status!r}")
            counts["incomplete"] += 1
            continue
        if status == "incomplete":
            raise ValueError(f"{key} complete measurements cannot be marked incomplete")
        _verify_aggregate_from_measurements(
            aggregate,
            measurements_by_key.get(str(key), []),
            stimulus_ids,
            repeats,
        )

        median_lufs = _finite_number(aggregate.get("median_lufs"), f"{key}.median_lufs")
        mad_lu = _finite_number(aggregate.get("mad_lu"), f"{key}.mad_lu")
        gain_db = _finite_number(aggregate.get("gain_db"), f"{key}.gain_db")
        _finite_number(aggregate.get("stimulus_spread_lu"), f"{key}.stimulus_spread_lu")
        _finite_number(aggregate.get("max_raw_peak"), f"{key}.max_raw_peak")
        predicted_peak = _finite_number(aggregate.get("predicted_peak"), f"{key}.predicted_peak")
        stimulus_medians = aggregate.get("stimulus_medians")
        repeat_mads = aggregate.get("repeat_mad_by_stimulus")
        if not isinstance(stimulus_medians, Mapping) or set(stimulus_medians) != {
            item["id"] for item in report["stimuli"]
        }:
            raise ValueError(f"{key}.stimulus_medians does not cover the declared corpus")
        if not isinstance(repeat_mads, Mapping) or set(repeat_mads) != set(stimulus_medians):
            raise ValueError(f"{key}.repeat_mad_by_stimulus does not cover the declared corpus")
        for stimulus_id, value in stimulus_medians.items():
            _finite_number(value, f"{key}.{stimulus_id}.median_lufs")
        for stimulus_id, value in repeat_mads.items():
            _finite_number(value, f"{key}.{stimulus_id}.mad_lu")
        if mad_lu < 0 or any(float(value) < 0 for value in repeat_mads.values()):
            raise ValueError(f"{key} has a negative repeat MAD")

        override = identity_overrides.get(str(key))
        minimum_gain = (
            float(override["min_gain_db"]) if override is not None else float(policy["min_gain_db"])
        )
        maximum_gain = float(policy["max_gain_db"])
        requested_gain = reference_lufs - median_lufs
        expected_gain = min(maximum_gain, max(minimum_gain, requested_gain))
        if not math.isclose(gain_db, expected_gain, abs_tol=1e-9):
            raise ValueError(f"{key}.gain_db does not match the declared policy")
        expected_limited = math.isclose(gain_db, minimum_gain, abs_tol=1e-12) or math.isclose(
            gain_db, maximum_gain, abs_tol=1e-12
        )
        if aggregate.get("gain_limited") is not expected_limited:
            raise ValueError(f"{key}.gain_limited does not match the declared policy")
        if expected_limited and not allow_gain_limited:
            raise ValueError(f"{key} gain hits a policy bound; explicit review is required")
        high_variability = mad_lu > float(policy["max_mad_lu"])
        if (status == "high_variability") != high_variability:
            raise ValueError(f"{key} variability status does not match its repeat MAD")
        peak_risk = gain_db > 0 and predicted_peak > 1.0
        if aggregate.get("peak_safety_review_required") is not peak_risk:
            raise ValueError(f"{key} peak safety flag does not match the predicted peak")
        if peak_risk and not allow_peak_risk:
            counts["high_variability"] += int(high_variability)
            if high_variability:
                continue
            raise ValueError(
                f"{key} positive gain predicts clipping; explicit peak-risk review required"
            )

        if high_variability:
            counts["high_variability"] += 1
            if not include_high_variability:
                continue
        else:
            counts["eligible"] += 1

        voices[str(key)] = {
            "gain_db": gain_db,
            "measured_lufs": median_lufs,
            "reference_lufs": reference_lufs,
            "mad_lu": mad_lu,
            "samples": repeat_count,
            "method": "bs1770",
            "corpus_version": corpus,
        }

    if not voices:
        raise ValueError("no eligible voice measurements to promote")
    candidate = {
        "schema": 1,
        "method": "bs1770",
        "corpus": corpus,
        "reference_lufs": reference_lufs,
        "generated_with": dict(sorted(generated_with.items())),
        "voices": dict(sorted(voices.items())),
    }
    return candidate, counts


def build_candidate_from_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    allow_partial: bool = False,
    include_high_variability: bool = False,
    allow_peak_risk: bool = False,
    allow_gain_limited: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Validate and merge disjoint candidate catalogs from several reports."""
    if not reports:
        raise ValueError("at least one measurement report is required")

    candidate: dict[str, Any] | None = None
    counts = {"eligible": 0, "high_variability": 0, "incomplete": 0}
    metadata_fields = ("schema", "method", "corpus", "reference_lufs", "generated_with")
    for report in reports:
        next_candidate, next_counts = build_candidate(
            report,
            allow_partial=allow_partial,
            include_high_variability=include_high_variability,
            allow_peak_risk=allow_peak_risk,
            allow_gain_limited=allow_gain_limited,
        )
        if candidate is None:
            candidate = next_candidate
        else:
            for field in metadata_fields:
                if candidate[field] != next_candidate[field]:
                    raise ValueError(f"measurement reports have mismatched {field}")
            duplicates = set(candidate["voices"]) & set(next_candidate["voices"])
            if duplicates:
                raise ValueError(
                    f"duplicate calibration key across reports: {sorted(duplicates)[0]}"
                )
            candidate["voices"].update(next_candidate["voices"])
        for name, count in next_counts.items():
            counts[name] += count

    if candidate is None:
        raise ValueError("at least one measurement report is required")
    candidate["voices"] = dict(sorted(candidate["voices"].items()))
    return candidate, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="*", type=Path, help="measurement report JSON files to merge"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--include-high-variability", action="store_true")
    parser.add_argument("--allow-peak-risk", action="store_true")
    parser.add_argument("--allow-gain-limited", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_paths = args.reports or [DEFAULT_INPUT]
    output = args.output.resolve()
    if output == PRODUCTION_CATALOG.resolve():
        raise SystemExit("promotion never writes directly to the packaged production catalog")
    if any(output == report_path.resolve() for report_path in report_paths):
        raise SystemExit("candidate output must not overwrite a measurement report")
    try:
        reports = []
        for report_path in report_paths:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, Mapping):
                raise ValueError(f"measurement report must contain a JSON object: {report_path}")
            reports.append(report)
        candidate, counts = build_candidate_from_reports(
            reports,
            allow_partial=args.allow_partial,
            include_high_variability=args.include_high_variability,
            allow_peak_risk=args.allow_peak_risk,
            allow_gain_limited=args.allow_gain_limited,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"cannot build calibration candidate: {error}") from error
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Eligible: {counts['eligible']}")
    print(f"High variability: {counts['high_variability']}")
    print(f"Incomplete: {counts['incomplete']}")
    print(f"Candidate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
