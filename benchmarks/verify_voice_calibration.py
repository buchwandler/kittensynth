#!/usr/bin/env python3
"""Verify packaged static calibration against real post-gain Kitten synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from audiosig import measure_loudness

from kittensynth import DEFAULT_VOICE_ALIASES, KittenVoice, SynthesisConfig, VoiceLevelConfig
from kittensynth.voice_level import VoiceCalibrationKey, load_voice_calibration

if __package__:
    from . import voice_level_benchmark
else:
    import voice_level_benchmark

POLICY_PATH = Path(__file__).with_name("data") / "voice_calibration_verification_policy.json"
MEASUREMENT_POLICY_PATH = Path(__file__).with_name("data") / "voice_level_policy.json"
STIMULI_PATH = voice_level_benchmark.STIMULI_PATH
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/verification.json")
CATALOG_RESOURCE = files("kittensynth").joinpath("data").joinpath("voice_level_calibration.json")
EXPECTED_MODELS = ("micro-0.8", "mini-0.8", "nano-0.8-int8", "nano-0.8-fp32")
_POLICY_FIELDS = {
    "schema",
    "name",
    "reference_lufs",
    "max_post_calibration_abs_error_lu",
    "max_post_calibration_peak_dbfs",
    "models",
    "voices",
}
_EPSILON = 1e-9


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_verification_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load the versioned release scope and calibrated acceptance limits."""
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
        raise ValueError("calibrated verification policy has an invalid shape")
    schema = policy.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError("calibrated verification policy has an unsupported schema")
    if policy.get("name") != "kittensynth-calibrated-verification-v1":
        raise ValueError("calibrated verification policy has an unsupported name")
    reference_lufs = _finite(policy.get("reference_lufs"), "reference_lufs")
    max_error = _finite(
        policy.get("max_post_calibration_abs_error_lu"),
        "max_post_calibration_abs_error_lu",
    )
    max_peak_dbfs = _finite(
        policy.get("max_post_calibration_peak_dbfs"), "max_post_calibration_peak_dbfs"
    )
    if max_error < 0 or max_peak_dbfs > 0:
        raise ValueError("calibrated verification policy limits are invalid")
    models = policy.get("models")
    voices = policy.get("voices")
    if not isinstance(models, list) or tuple(models) != EXPECTED_MODELS:
        raise ValueError("calibrated verification policy must cover the four promoted models")
    if not isinstance(voices, list) or tuple(voices) != tuple(DEFAULT_VOICE_ALIASES):
        raise ValueError("calibrated verification policy must cover all eight promoted voices")
    if any(not isinstance(value, str) for value in (*models, *voices)):
        raise ValueError("calibrated verification model and voice IDs must be strings")
    return {
        **dict(policy),
        "reference_lufs": reference_lufs,
        "max_post_calibration_abs_error_lu": max_error,
        "max_post_calibration_peak_dbfs": max_peak_dbfs,
    }


def _expected_identities(policy: Mapping[str, Any], catalog: Any) -> list[dict[str, Any]]:
    identities = []
    for model_id in policy["models"]:
        for alias in policy["voices"]:
            internal_voice = DEFAULT_VOICE_ALIASES[alias]
            key = VoiceCalibrationKey("kitten", model_id, internal_voice)
            calibration = catalog.voices.get(key)
            identities.append(
                {
                    "model_id": model_id,
                    "voice_alias": alias,
                    "internal_voice": internal_voice,
                    "calibration_key": str(key),
                    "expected_gain_db": calibration.gain_db if calibration is not None else None,
                    "status": "pending",
                    "complete": False,
                    "stimulus_medians_lufs": {},
                    "median_lufs": None,
                    "abs_error_lu": None,
                    "max_sample_peak_dbfs": None,
                    "measurements": [],
                    "issues": [],
                }
            )
    return identities


def _add_failure(
    identity: dict[str, Any],
    failures: list[dict[str, Any]],
    *,
    phase: str,
    message: str,
    error_type: str = "VerificationFailure",
    stimulus_id: str | None = None,
    repeat: int | None = None,
) -> None:
    failure = {
        "model_id": identity["model_id"],
        "voice_alias": identity["voice_alias"],
        "internal_voice": identity["internal_voice"],
        "calibration_key": identity["calibration_key"],
        "phase": phase,
        "error_type": error_type,
        "error": message,
    }
    if stimulus_id is not None:
        failure["stimulus_id"] = stimulus_id
    if repeat is not None:
        failure["repeat"] = repeat
    failures.append(failure)
    identity["issues"].append(message)


def _check_runtime_metadata(
    result: Any,
    *,
    identity: Mapping[str, Any],
    catalog_revision: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None, ["synthesis result metadata is missing"]
    voice_level = metadata.get("voice_level")
    if not isinstance(voice_level, Mapping):
        return None, ["synthesis result voice_level metadata is missing"]
    checks = {
        "mode": "calibrated",
        "source": "catalog",
        "calibration_key": identity["calibration_key"],
        "catalog_revision": catalog_revision,
    }
    issues = [
        f"voice_level.{field} expected {expected!r}, got {voice_level.get(field)!r}"
        for field, expected in checks.items()
        if voice_level.get(field) != expected
    ]
    gain = voice_level.get("gain_db")
    try:
        actual_gain = _finite(gain, "voice_level.gain_db")
    except ValueError:
        issues.append("voice_level.gain_db is missing or non-finite")
        actual_gain = None
    if actual_gain is not None and not math.isclose(
        actual_gain, float(identity["expected_gain_db"]), rel_tol=0.0, abs_tol=1e-12
    ):
        issues.append(
            f"voice_level.gain_db expected {identity['expected_gain_db']!r}, got {actual_gain!r}"
        )
    actual = {
        "mode": voice_level.get("mode"),
        "source": voice_level.get("source"),
        "calibration_key": voice_level.get("calibration_key"),
        "catalog_revision": voice_level.get("catalog_revision"),
        "gain_db": actual_gain,
    }
    return actual, issues


def _measure_identity(
    model: Any,
    identity: dict[str, Any],
    stimuli: Sequence[Mapping[str, str]],
    repeats: int,
    policy: Mapping[str, Any],
    catalog_revision: str,
    failures: list[dict[str, Any]],
) -> None:
    values_by_stimulus: dict[str, list[float]] = defaultdict(list)
    peaks_dbfs: list[float] = []
    config = SynthesisConfig(speed=1.0, voice_level=VoiceLevelConfig(mode="calibrated"))

    for stimulus in stimuli:
        for repeat in range(repeats):
            row: dict[str, Any] = {
                "stimulus_id": stimulus["id"],
                "repeat": repeat,
                "status": "failed",
            }
            try:
                result = model.synthesize_prepared(
                    stimulus["text"], voice=identity["voice_alias"], config=config
                )
            except Exception as error:
                row.update(
                    {
                        "phase": "synthesis",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                _add_failure(
                    identity,
                    failures,
                    phase="synthesis",
                    message=str(error),
                    error_type=type(error).__name__,
                    stimulus_id=stimulus["id"],
                    repeat=repeat,
                )
                identity["measurements"].append(row)
                continue

            voice_metadata, metadata_issues = _check_runtime_metadata(
                result, identity=identity, catalog_revision=catalog_revision
            )
            row["voice_level"] = voice_metadata
            for message in metadata_issues:
                _add_failure(
                    identity,
                    failures,
                    phase="metadata",
                    message=message,
                    stimulus_id=stimulus["id"],
                    repeat=repeat,
                )
            try:
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if not audio.size or not np.all(np.isfinite(audio)):
                    raise ValueError("synthesis audio must be non-empty and finite")
                sample_rate = int(result.sample_rate)
                if sample_rate <= 0:
                    raise ValueError("sample rate must be positive")
                peak = float(np.max(np.abs(audio)))
                if peak <= 0:
                    raise ValueError("sample peak must be positive")
                peak_dbfs = 20.0 * math.log10(peak)
                row.update(
                    {
                        "sample_rate": sample_rate,
                        "sample_peak": peak,
                        "sample_peak_dbfs": peak_dbfs,
                    }
                )
                peaks_dbfs.append(peak_dbfs)
            except Exception as error:
                row.update(
                    {
                        "phase": "measurement",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                _add_failure(
                    identity,
                    failures,
                    phase="measurement",
                    message=str(error),
                    error_type=type(error).__name__,
                    stimulus_id=stimulus["id"],
                    repeat=repeat,
                )
                identity["measurements"].append(row)
                continue

            try:
                loudness = measure_loudness(audio, sample_rate=sample_rate)
                integrated_lufs = loudness.integrated_lufs
                if integrated_lufs is None:
                    raise ValueError("integrated loudness is unavailable")
                integrated_lufs = _finite(integrated_lufs, "integrated_lufs")
            except Exception as error:
                row.update(
                    {
                        "phase": "measurement",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                _add_failure(
                    identity,
                    failures,
                    phase="measurement",
                    message=str(error),
                    error_type=type(error).__name__,
                    stimulus_id=stimulus["id"],
                    repeat=repeat,
                )
                identity["measurements"].append(row)
                continue

            row.update(
                {
                    "integrated_lufs": integrated_lufs,
                    "status": "passed" if not metadata_issues else "failed",
                }
            )
            values_by_stimulus[stimulus["id"]].append(integrated_lufs)
            identity["measurements"].append(row)

    expected_samples = len(stimuli) * repeats
    identity["complete"] = len(identity["measurements"]) == expected_samples and all(
        len(values_by_stimulus[stimulus["id"]]) == repeats for stimulus in stimuli
    )
    if identity["complete"]:
        stimulus_medians = {
            stimulus["id"]: statistics.median(values_by_stimulus[stimulus["id"]])
            for stimulus in stimuli
        }
        median_lufs = statistics.median(stimulus_medians.values())
        abs_error = abs(median_lufs - float(policy["reference_lufs"]))
        identity["stimulus_medians_lufs"] = stimulus_medians
        identity["median_lufs"] = median_lufs
        identity["abs_error_lu"] = abs_error
        if abs_error > float(policy["max_post_calibration_abs_error_lu"]) + _EPSILON:
            _add_failure(
                identity,
                failures,
                phase="acceptance",
                message=(
                    f"absolute median loudness error {abs_error:.6f} LU exceeds "
                    f"{policy['max_post_calibration_abs_error_lu']:.6f} LU"
                ),
            )
    else:
        _add_failure(
            identity,
            failures,
            phase="coverage",
            message=(
                "identity has incomplete measurements: "
                f"{len(identity['measurements'])}/{expected_samples}"
            ),
        )

    if peaks_dbfs:
        max_peak_dbfs = max(peaks_dbfs)
        identity["max_sample_peak_dbfs"] = max_peak_dbfs
        if max_peak_dbfs > float(policy["max_post_calibration_peak_dbfs"]) + _EPSILON:
            _add_failure(
                identity,
                failures,
                phase="acceptance",
                message=(
                    f"maximum sample peak {max_peak_dbfs:.6f} dBFS exceeds "
                    f"{policy['max_post_calibration_peak_dbfs']:.6f} dBFS"
                ),
            )

    identity["status"] = "passed" if not identity["issues"] and identity["complete"] else "failed"


def _resource_bytes(resource: Any) -> bytes:
    return resource.read_bytes()


def run_verification(
    *,
    open_model: Callable[..., Any] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    quality: str | None = None,
    refresh_catalog: bool = False,
) -> dict[str, Any]:
    """Run calibrated acceptance checks and return a complete evidence report."""
    policy = load_verification_policy()
    measurement_policy = voice_level_benchmark.load_policy(MEASUREMENT_POLICY_PATH)
    if policy["reference_lufs"] != measurement_policy["reference_lufs"]:
        raise ValueError("verification reference_lufs differs from the raw calibration policy")
    stimulus_document = voice_level_benchmark.load_stimuli(STIMULI_PATH)
    stimuli = stimulus_document["stimuli"]
    catalog_bytes = _resource_bytes(CATALOG_RESOURCE)
    catalog = load_voice_calibration(CATALOG_RESOURCE)
    identities = _expected_identities(policy, catalog)
    failures: list[dict[str, Any]] = []

    for identity in identities:
        if identity["expected_gain_db"] is None:
            _add_failure(
                identity,
                failures,
                phase="catalog",
                message="packaged catalog has no exact entry for this identity",
            )

    factory = open_model or KittenVoice.from_pretrained
    for model_id in policy["models"]:
        model_id_identities = [
            identity for identity in identities if identity["model_id"] == model_id
        ]
        try:
            model = factory(
                model_id,
                quality=quality,
                cache_dir=cache_dir,
                offline=offline,
                refresh_catalog=refresh_catalog,
            )
        except Exception as error:
            for identity in model_id_identities:
                _add_failure(
                    identity,
                    failures,
                    phase="model_open",
                    message=str(error),
                    error_type=type(error).__name__,
                )
            continue

        try:
            if model.model_id != model_id:
                for identity in model_id_identities:
                    _add_failure(
                        identity,
                        failures,
                        phase="model_identity",
                        message=f"opened model ID {model.model_id!r} does not match {model_id!r}",
                    )
                continue
            available = set(model.available_voices)
            for identity in model_id_identities:
                if identity["expected_gain_db"] is None:
                    continue
                alias = identity["voice_alias"]
                if alias not in available:
                    _add_failure(
                        identity,
                        failures,
                        phase="voice_identity",
                        message=f"expected voice alias {alias!r} is unavailable",
                    )
                    continue
                try:
                    actual_internal_voice = model.voice_bank.resolve(alias)
                except Exception as error:
                    _add_failure(
                        identity,
                        failures,
                        phase="voice_identity",
                        message=str(error),
                        error_type=type(error).__name__,
                    )
                    continue
                if actual_internal_voice != identity["internal_voice"]:
                    _add_failure(
                        identity,
                        failures,
                        phase="voice_identity",
                        message=(
                            f"voice {alias!r} resolved to {actual_internal_voice!r}, expected "
                            f"{identity['internal_voice']!r}"
                        ),
                    )
                    continue
                _measure_identity(
                    model,
                    identity,
                    stimuli,
                    int(measurement_policy["repeats"]),
                    policy,
                    str(catalog.revision),
                    failures,
                )
        except Exception as error:
            for identity in model_id_identities:
                if identity["status"] == "pending":
                    _add_failure(
                        identity,
                        failures,
                        phase="verification",
                        message=str(error),
                        error_type=type(error).__name__,
                    )
        finally:
            try:
                model.close()
            except Exception as error:
                for identity in model_id_identities:
                    _add_failure(
                        identity,
                        failures,
                        phase="model_close",
                        message=str(error),
                        error_type=type(error).__name__,
                    )

    for identity in identities:
        if identity["status"] == "pending" and not identity["issues"]:
            _add_failure(
                identity,
                failures,
                phase="verification",
                message="identity was not verified",
            )
        identity["status"] = (
            "passed" if identity["complete"] and not identity["issues"] else "failed"
        )
    measured = sum(1 for identity in identities if identity["complete"])
    passed = sum(1 for identity in identities if identity["status"] == "passed")
    return {
        "schema": 1,
        "status": "passed" if passed == len(identities) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"name": "kittensynth-calibrated-verification", "schema": 1},
        "generated_with": {
            **voice_level_benchmark.generated_with(),
            "python": platform.python_version(),
        },
        "catalog": {
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "revision": catalog.revision,
            "schema": catalog.schema,
            "corpus": catalog.corpus,
            "reference_lufs": catalog.reference_lufs,
        },
        "policy": policy,
        "measurement_policy": {
            "schema": measurement_policy["schema"],
            "name": measurement_policy["name"],
            "repeats": measurement_policy["repeats"],
        },
        "stimuli": {
            "schema": stimulus_document["schema"],
            "name": stimulus_document["name"],
            "language": stimulus_document["language"],
            "sha256": hashlib.sha256(STIMULI_PATH.read_bytes()).hexdigest(),
            "items": stimuli,
        },
        "coverage": {
            "identities_expected": len(identities),
            "identities_measured": measured,
            "identities_passed": passed,
            "identities_failed": len(identities) - passed,
            "measurement_failures": sum(
                1 for failure in failures if failure["phase"] in {"synthesis", "measurement"}
            ),
            "failure_count": len(failures),
            "complete": measured == len(identities),
        },
        "identities": identities,
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_verification(
            cache_dir=args.cache_dir,
            offline=args.offline,
            quality=args.quality,
            refresh_catalog=args.refresh_catalog,
        )
    except Exception as error:
        print(f"Calibrated verification could not start: {type(error).__name__}: {error}")
        return 1
    voice_level_benchmark.write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Report: {args.output}")
    print(
        f"Coverage: {coverage['identities_passed']}/{coverage['identities_expected']} passed; "
        f"{coverage['measurement_failures']} failure(s)"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
