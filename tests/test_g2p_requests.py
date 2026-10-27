from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kittensynth import (
    KittenG2P,
    KittenG2PConfig,
    KittenVoice,
    LinguisticToken,
    PronunciationOverride,
)


@dataclass
class InferenceResult:
    audio: np.ndarray
    sample_rate: int = 24000


class FakeRuntime:
    def __init__(self) -> None:
        self.inputs: list[tuple[int, ...]] = []

    def infer(self, token_ids: tuple[int, ...], **kwargs: Any) -> InferenceResult:
        self.inputs.append(tuple(token_ids))
        return InferenceResult(np.array([0.0, 0.25, -0.25], dtype=np.float32))

    def close(self) -> None:
        pass


class TextBackend:
    def phonemize(self, text: str, *, language: str) -> str:
        assert language == "en-us"
        return "həloʊ"

    def close(self) -> None:
        pass


class RecordingG2P:
    def __init__(self) -> None:
        self.frontend = KittenG2P(backend=TextBackend())
        self.calls: list[tuple[str, Any, Any]] = []

    def phonemize_prepared(
        self,
        text: str,
        *,
        tokens: Any = None,
        pronunciation_overrides: Any = None,
    ):
        self.calls.append((text, tokens, pronunciation_overrides))
        return self.frontend.phonemize_prepared(
            text, tokens=tokens, pronunciation_overrides=pronunciation_overrides
        )

    def close(self) -> None:
        self.frontend.close()


class LegacyG2P:
    """Pre-request-option injection shape: only accepts a text argument."""

    def __init__(self) -> None:
        self.frontend = KittenG2P(backend=TextBackend())

    def phonemize_prepared(self, text: str):
        return self.frontend.phonemize_prepared(text)

    def close(self) -> None:
        self.frontend.close()


def make_voice(tmp_path: Path, *, g2p: Any, g2p_config: KittenG2PConfig | None = None):
    voices_path = tmp_path / "voices.npz"
    np.savez(voices_path, **{"expr-voice-2-m": np.arange(40, dtype=np.float32).reshape(10, 4)})
    runtime = FakeRuntime()
    voice = KittenVoice(
        runtime=runtime,
        voices_path=voices_path,
        metadata={"voice_aliases": {"Jasper": "expr-voice-2-m"}},
        model_ref=None,
        g2p=g2p,
        g2p_config=g2p_config,
    )
    return voice, runtime


def test_measurement_and_synthesis_forward_identical_tokens_and_overrides(tmp_path):
    text = "Hello, bow!"
    start = text.index("bow")
    tokens = (LinguisticToken(start, start + 3, "bow", pos="VERB", tag="VB"),)
    overrides = (PronunciationOverride(start, start + 3, "baʊ"),)
    g2p = RecordingG2P()
    voice, runtime = make_voice(tmp_path, g2p=g2p)

    measured = voice.measure_prepared(text, tokens=tokens, pronunciation_overrides=overrides)
    assert runtime.inputs == []
    result = voice.synthesize_prepared(text, tokens=tokens, pronunciation_overrides=overrides)

    assert measured.token_count == len(runtime.inputs[-1])
    assert g2p.calls == [(text, tokens, overrides), (text, tokens, overrides)]
    assert result.metadata is not None
    assert result.metadata["pronunciation_trace"][-1]["source"] == "explicit"
    assert result.metadata["pronunciation_trace"][-1]["text"] == "bow"
    assert result.metadata["pronunciation_diagnostics"] == []
    voice.close()


def test_legacy_injected_g2p_keeps_one_argument_call_compatible(tmp_path):
    g2p = LegacyG2P()
    voice, _runtime = make_voice(tmp_path, g2p=g2p)

    assert voice.measure_prepared("hello").token_count > 0
    voice.close()


def test_g2p_config_is_forwarded_and_identified_in_diagnostics(tmp_path, monkeypatch):
    config = KittenG2PConfig(lexicons=("lexhint",), heteronym_policy="off")
    created: list[KittenG2PConfig | None] = []

    class ConfiguredG2P(LegacyG2P):
        def __init__(self, config: KittenG2PConfig | None = None) -> None:
            super().__init__()
            self.config = config
            created.append(config)

    monkeypatch.setattr("kittensynth.voice.KittenG2P", ConfiguredG2P)
    voice, _runtime = make_voice(tmp_path, g2p=None, g2p_config=config)

    assert created == [config]
    identity = voice.diagnostics()["g2p_identity"]
    assert identity["config"]["lexicons"] == ["lexhint"]
    assert identity["config"]["heteronym_policy"] == "off"
    voice.close()


def test_custom_g2p_and_g2p_config_are_rejected_as_ambiguous(tmp_path):
    with pytest.raises(ValueError, match="cannot both be supplied"):
        make_voice(
            tmp_path,
            g2p=LegacyG2P(),
            g2p_config=KittenG2PConfig(),
        )
