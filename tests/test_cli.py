from __future__ import annotations

from typing import Any

import pytest

from kittensynth import __main__ as cli


class FakeResult:
    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_wav(self, path: str) -> None:
        self.saved_to = path


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = FakeResult()

    def __enter__(self) -> FakeModel:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> FakeResult:
        self.calls.append({"text": text, "voice": voice, "config": config})
        return self.result


def run_cli(monkeypatch, argv: list[str]) -> FakeModel:
    model = FakeModel()
    opened: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(model_id: str, **kwargs: Any) -> FakeModel:
        opened.append((model_id, kwargs))
        return model

    monkeypatch.setattr(cli.KittenVoice, "from_pretrained", from_pretrained)
    assert cli.main(argv) == 0
    assert opened
    return model


def test_cli_defaults_to_calibration_off(monkeypatch, capsys):
    model = run_cli(monkeypatch, ["Prepared text", "--output", "out.wav"])

    assert model.calls[0]["config"].speed == 1.0
    assert model.calls[0]["config"].voice_level.mode == "off"
    assert model.calls[0]["config"].voice_level.gain_db is None
    assert model.result.saved_to == "out.wav"
    assert capsys.readouterr().out.strip() == "out.wav"


def test_cli_accepts_calibration_mode_gain_and_speed(monkeypatch):
    model = run_cli(
        monkeypatch,
        [
            "Prepared text",
            "--speed",
            "0.9",
            "--voice-level",
            "calibrated",
            "--gain-db",
            "2.5",
        ],
    )

    config = model.calls[0]["config"]
    assert config.speed == 0.9
    assert config.voice_level.mode == "calibrated"
    assert config.voice_level.gain_db == 2.5


def test_cli_rejects_unknown_voice_level_mode():
    with pytest.raises(SystemExit):
        cli.parse_args(["Prepared text", "--voice-level", "automatic"])
