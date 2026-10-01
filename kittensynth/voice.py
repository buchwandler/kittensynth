from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from kitteng2p import KittenG2P

from ._onnxvoice import (
    install_pretrained_model,
    load_local_config,
    open_installed_model,
    open_local_model,
)
from .config import SynthesisConfig
from .errors import EmptyTextError
from .types import SynthesisResult, VoiceInfo
from .voice_bank import VoiceBank


class KittenVoice:
    def __init__(
        self,
        *,
        runtime: Any,
        voices_path: str | Path,
        metadata: Mapping[str, Any],
        model_ref: str | None,
        g2p: KittenG2P | None = None,
        sample_rate: int = 24000,
    ) -> None:
        self.runtime = runtime
        self.model_ref = model_ref
        self.metadata = dict(metadata)
        self.sample_rate = int(sample_rate)
        self._owns_g2p = g2p is None
        self.g2p = g2p or KittenG2P()
        self.voice_bank = VoiceBank(
            voices_path,
            voice_aliases=self.metadata.get("voice_aliases"),
            speed_priors=self.metadata.get("speed_priors"),
        )
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model: str = "nano-0.8-int8",
        *,
        quality: str | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh_catalog: bool = False,
        force_download: bool = False,
        catalog_url: str | None = None,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        progress: Any | None = None,
        g2p: KittenG2P | None = None,
    ) -> KittenVoice:
        resolved = install_pretrained_model(
            model,
            quality=quality,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            catalog_url=catalog_url,
            progress=progress,
        )
        runtime = open_installed_model(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            cache_dir=cache_dir,
            offline=offline,
            catalog_url=catalog_url,
        )
        return cls(
            runtime=runtime,
            voices_path=resolved.voices_path,
            metadata=resolved.metadata,
            model_ref=resolved.ref,
            g2p=g2p,
            sample_rate=resolved.sample_rate,
        )

    @classmethod
    def from_local(
        cls,
        *,
        model_path: str | Path,
        voices_path: str | Path,
        config_path: str | Path | None = None,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        g2p: KittenG2P | None = None,
    ) -> KittenVoice:
        metadata = load_local_config(config_path)
        runtime = open_local_model(
            model_path=model_path,
            voices_path=voices_path,
            metadata=metadata,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(
            runtime=runtime,
            voices_path=voices_path,
            metadata=metadata,
            model_ref=None,
            g2p=g2p,
            sample_rate=int(metadata.get("sample_rate") or 24000),
        )

    @property
    def available_voices(self) -> tuple[str, ...]:
        return self.voice_bank.available_voices

    @property
    def voices(self) -> tuple[VoiceInfo, ...]:
        return tuple(
            VoiceInfo(name=name, internal_id=self.voice_bank.resolve(name))
            for name in self.available_voices
        )

    def synthesize_prepared(
        self,
        text: str,
        *,
        voice: str = "Jasper",
        speed: float = 1.0,
        config: SynthesisConfig | None = None,
    ) -> SynthesisResult:
        if self._closed:
            raise RuntimeError("KittenVoice is closed")
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("text must contain speakable content")
        if config is not None:
            speed = config.validated().speed

        frontend = self.g2p.phonemize_prepared(text)
        style = self.voice_bank.style_for(voice, text_length=len(text))
        effective_speed = self.voice_bank.effective_speed(voice, speed)

        result = self.runtime.infer(
            frontend.token_ids,
            style=style,
            speed=effective_speed,
        )
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            voice=voice,
            model_ref=self.model_ref,
            speed=effective_speed,
            metadata={
                "phonemes": frontend.phonemes,
                "token_count": len(frontend.token_ids),
                "dropped_symbols": frontend.dropped_symbols,
                "internal_voice": self.voice_bank.resolve(voice),
            },
        )

    # Small compatibility alias for callers migrating from the first MVP.
    synthesize_text = synthesize_prepared

    def diagnostics(self) -> dict[str, object]:
        runtime_diagnostics = getattr(self.runtime, "diagnostics", None)
        g2p_diagnostics = getattr(getattr(self.g2p, "backend", None), "diagnostics", None)
        return {
            "runtime": runtime_diagnostics()
            if callable(runtime_diagnostics)
            else runtime_diagnostics,
            "g2p": g2p_diagnostics,
        }

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()
        if self._owns_g2p:
            self.g2p.close()
        self._closed = True

    def __enter__(self) -> KittenVoice:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
