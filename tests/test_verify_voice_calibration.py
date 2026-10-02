from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import verify_voice_calibration as verifier
from kittensynth import DEFAULT_VOICE_ALIASES
from kittensynth.voice_level import VoiceCalibrationKey, load_voice_calibration


class FakeVoiceBank:
    def resolve(self, alias: str) -> str:
        return DEFAULT_VOICE_ALIASES[alias]


class FakeModel:
    def __init__(
        self,
        model_id: str,
        *,
        peak: float = 0.5,
        metadata_source: str = "catalog",
    ) -> None:
        self.model_id = model_id
        self.available_voices = tuple(DEFAULT_VOICE_ALIASES)
        self.voice_bank = FakeVoiceBank()
        self.peak = peak
        self.metadata_source = metadata_source
        self.calls: list[dict[str, Any]] = []
        self.catalog = load_voice_calibration(verifier.CATALOG_RESOURCE)

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
        assert config.voice_level.mode == "calibrated"
        key = f"kitten:{self.model_id}:{self.voice_bank.resolve(voice)}"
        calibration = self.catalog.voices[VoiceCalibrationKey.parse(key)]
        self.calls.append({"text": text, "voice": voice, "config": config})
        return SimpleNamespace(
            audio=np.full(128, self.peak, dtype=np.float32),
            sample_rate=24000,
            metadata={
                "voice_level": {
                    "mode": "calibrated",
                    "source": self.metadata_source,
                    "calibration_key": key,
                    "catalog_revision": self.catalog.revision,
                    "gain_db": calibration.gain_db,
                }
            },
        )

    def close(self) -> None:
        pass


def factory_for(
    *, peak: float = 0.5, metadata_source: str = "catalog", fail_model: str | None = None
):
    opened: list[FakeModel] = []

    def open_model(model_id: str, **kwargs: Any) -> FakeModel:
        if model_id == fail_model:
            raise RuntimeError("synthetic model-open failure")
        model = FakeModel(model_id, peak=peak, metadata_source=metadata_source)
        opened.append(model)
        return model

    return open_model, opened


def test_verification_policy_has_user_approved_release_limits():
    policy = verifier.load_verification_policy()

    assert policy["reference_lufs"] == -24.0
    assert policy["max_post_calibration_abs_error_lu"] == 0.5
    assert policy["max_post_calibration_peak_dbfs"] == -1.0
    assert len(policy["models"]) == 4
    assert len(policy["voices"]) == 8


def test_verifier_passes_complete_mocked_32_identity_catalog(monkeypatch):
    open_model, opened = factory_for()
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-24.0),
    )

    report = verifier.run_verification(open_model=open_model)

    assert report["status"] == "passed"
    assert report["coverage"] == {
        "identities_expected": 32,
        "identities_measured": 32,
        "identities_passed": 32,
        "identities_failed": 0,
        "measurement_failures": 0,
        "failure_count": 0,
        "complete": True,
    }
    assert len(opened) == 4
    assert sum(len(model.calls) for model in opened) == 32 * 3 * 3
    assert all(identity["status"] == "passed" for identity in report["identities"])
    assert all(identity["abs_error_lu"] == 0.0 for identity in report["identities"])
    assert report["catalog"]["revision"]
    assert report["catalog"]["sha256"]
    assert report["generated_with"]["python"]


def test_verifier_fails_loudness_and_peak_policy_violations(monkeypatch):
    open_model, _ = factory_for(peak=0.95)
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-23.4),
    )

    report = verifier.run_verification(open_model=open_model)

    assert report["status"] == "failed"
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["identities_passed"] == 0
    assert report["coverage"]["identities_failed"] == 32
    assert report["coverage"]["measurement_failures"] == 0
    assert {failure["phase"] for failure in report["failures"]} == {"acceptance"}
    assert all(identity["abs_error_lu"] == pytest.approx(0.6) for identity in report["identities"])
    assert all(identity["max_sample_peak_dbfs"] > -1.0 for identity in report["identities"])


def test_verifier_records_open_failures_without_dropping_identities(monkeypatch):
    open_model, _ = factory_for(fail_model="micro-0.8")
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-24.0),
    )

    report = verifier.run_verification(open_model=open_model)

    assert report["status"] == "failed"
    assert report["coverage"]["identities_expected"] == 32
    assert report["coverage"]["identities_measured"] == 24
    assert report["coverage"]["identities_passed"] == 24
    assert report["coverage"]["identities_failed"] == 8
    assert report["coverage"]["complete"] is False
    assert len([row for row in report["failures"] if row["phase"] == "model_open"]) == 8
    assert len(report["identities"]) == 32


def test_verifier_rejects_runtime_metadata_that_does_not_select_catalog(monkeypatch):
    open_model, _ = factory_for(metadata_source="override")
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-24.0),
    )

    report = verifier.run_verification(open_model=open_model)

    assert report["status"] == "failed"
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["identities_passed"] == 0
    assert all(
        any("voice_level.source" in issue for issue in identity["issues"])
        for identity in report["identities"]
    )


def test_cli_returns_nonzero_for_failed_verification(monkeypatch, tmp_path):
    report = {
        "status": "failed",
        "coverage": {
            "identities_passed": 31,
            "identities_expected": 32,
            "measurement_failures": 1,
        },
    }
    monkeypatch.setattr(verifier, "run_verification", lambda **kwargs: report)
    output = tmp_path / "report.json"

    result = verifier.main(["--output", str(output)])

    assert result == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_verifier_script_help_runs_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_voice_calibration.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--offline" in result.stdout
