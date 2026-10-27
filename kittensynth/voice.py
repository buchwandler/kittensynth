from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from kitteng2p import (
    KittenG2P,
    KittenG2PConfig,
    LinguisticToken,
    PronunciationOverride,
)

from ._onnxvoice import (
    install_pretrained_model,
    load_local_config,
    open_installed_model,
    open_local_model,
)
from .capacity import capacity_for_model
from .config import SynthesisConfig
from .errors import (
    EmptyTextError,
    KittenSynthError,
    ModelInferenceError,
    SynthesisInputTooLongError,
)
from .types import SynthesisMeasure, SynthesisResult, VoiceInfo
from .voice_bank import VoiceBank
from .voice_level import (
    VoiceCalibrationKey,
    VoiceLevelApplication,
    apply_voice_level_calibration,
)

TokenInput: TypeAlias = (
    Sequence[LinguisticToken | Mapping[str, Any]] | LinguisticToken | Mapping[str, Any] | None
)
PronunciationInput: TypeAlias = (
    Sequence[PronunciationOverride | Mapping[str, Any]]
    | PronunciationOverride
    | Mapping[str, Any]
    | None
)


def _validate_g2p_options(
    g2p: object | None,
    g2p_config: KittenG2PConfig | None,
) -> None:
    if g2p is not None and g2p_config is not None:
        raise ValueError("g2p and g2p_config cannot both be supplied")
    if g2p_config is not None and not isinstance(g2p_config, KittenG2PConfig):
        raise TypeError("g2p_config must be a KittenG2PConfig")
    if g2p_config is not None and "config" not in signature(KittenG2P).parameters:
        raise KittenSynthError(
            "g2p_config requires kitteng2p>=0.1.3; upgrade kitteng2p to enable G2Lex support"
        )


class KittenVoice:
    def __init__(
        self,
        *,
        runtime: Any,
        voices_path: str | Path,
        metadata: Mapping[str, Any],
        model_ref: str | None,
        g2p: KittenG2P | None = None,
        g2p_config: KittenG2PConfig | None = None,
        model_id: str | None = None,
        sample_rate: int = 24000,
    ) -> None:
        _validate_g2p_options(g2p, g2p_config)
        self.runtime = runtime
        self.model_ref = model_ref
        self.model_id = model_id
        self.metadata = dict(metadata)
        self._capacity = capacity_for_model(
            self.model_id,
            self.metadata.get("source_revision"),
        )
        self.sample_rate = int(sample_rate)
        self._owns_g2p = g2p is None
        if g2p is not None:
            self.g2p = g2p
        elif g2p_config is not None:
            self.g2p = KittenG2P(g2p_config)
        else:
            self.g2p = KittenG2P()
        self.g2p_config = getattr(self.g2p, "config", g2p_config)
        self.voice_bank = VoiceBank(
            voices_path,
            voice_aliases=self.metadata.get("voice_aliases"),
            speed_priors=self.metadata.get("speed_priors"),
        )
        self._closed = False
        self._last_voice_level_application: VoiceLevelApplication | None = None

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
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        progress: Any | None = None,
        g2p: KittenG2P | None = None,
        g2p_config: KittenG2PConfig | None = None,
    ) -> KittenVoice:
        _validate_g2p_options(g2p, g2p_config)
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
            model_id=resolved.model_id,
            g2p=g2p,
            g2p_config=g2p_config,
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
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        g2p: KittenG2P | None = None,
        g2p_config: KittenG2PConfig | None = None,
    ) -> KittenVoice:
        _validate_g2p_options(g2p, g2p_config)
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
            g2p_config=g2p_config,
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

    def calibration_key(self, voice: str) -> VoiceCalibrationKey | None:
        """Resolve a managed voice to its stable internal calibration identity."""
        if not self.model_id:
            return None
        internal_voice = self.voice_bank.resolve(voice)
        return VoiceCalibrationKey(
            model_source="kitten",
            model_id=self.model_id,
            voice=internal_voice,
        )

    @property
    def last_voice_level_application(self) -> VoiceLevelApplication | None:
        """Return the calibration decision made for the most recent synthesis."""
        return self._last_voice_level_application

    def _prepare_text(
        self,
        text: str,
        *,
        tokens: TokenInput = None,
        pronunciation_overrides: PronunciationInput = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("KittenVoice is closed")
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("text must contain speakable content")

        method = self.g2p.phonemize_prepared
        if tokens is None and pronunciation_overrides is None:
            # Preserve the historical one-argument protocol for injected G2Ps.
            return method(text)

        try:
            parameters = signature(method).parameters
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "the injected G2P does not expose token/override request capabilities"
            ) from exc
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        for name, value in (
            ("tokens", tokens),
            ("pronunciation_overrides", pronunciation_overrides),
        ):
            if value is None:
                continue
            if name not in parameters and not accepts_kwargs:
                raise TypeError(
                    f"the injected G2P does not support {name}; use KittenG2P "
                    "or a compatible frontend"
                )
            kwargs[name] = value
        return method(text, **kwargs)

    def _g2p_configuration_identity(self) -> dict[str, Any]:
        config = self.g2p_config
        selected_sources = getattr(self.g2p, "pronunciation_identity", ())
        selected = [
            {key: value for key, value in source.items() if key != "path"}
            for source in selected_sources
            if isinstance(source, Mapping)
        ]
        identity: dict[str, Any] = {
            "implementation": f"{type(self.g2p).__module__}.{type(self.g2p).__qualname__}",
            "selected_sources": selected,
        }
        if isinstance(config, KittenG2PConfig):
            identity["config"] = {
                "language": config.language,
                "espeak_mode": config.espeak_mode,
                "lexicons": list(config.lexicons) if config.lexicons is not None else None,
                "g2lex_files": [
                    {
                        "name": Path(reference.path).name,
                        "sha256": reference.sha256,
                        "logical_sha256": reference.logical_sha256,
                    }
                    for reference in config.g2lex_files
                ],
                "lexicon_data_policy": config.lexicon_data_policy,
                "heteronym_policy": config.heteronym_policy,
            }
        return identity

    def measure_prepared(
        self,
        text: str,
        *,
        tokens: TokenInput = None,
        pronunciation_overrides: PronunciationInput = None,
    ) -> SynthesisMeasure:
        """Measure exact Kitten model tokens without acoustic inference."""
        frontend = self._prepare_text(
            text, tokens=tokens, pronunciation_overrides=pronunciation_overrides
        )
        token_count = len(frontend.token_ids)
        return SynthesisMeasure(
            token_count=token_count,
            max_tokens=self._capacity.max_tokens,
            fits=token_count <= self._capacity.max_tokens,
            model_id=self.model_id,
            metadata={"capacity_policy_revision": self._capacity.policy_revision},
        )

    def synthesize_prepared(
        self,
        text: str,
        *,
        voice: str = "Jasper",
        speed: float = 1.0,
        config: SynthesisConfig | None = None,
        tokens: TokenInput = None,
        pronunciation_overrides: PronunciationInput = None,
    ) -> SynthesisResult:
        frontend = self._prepare_text(
            text, tokens=tokens, pronunciation_overrides=pronunciation_overrides
        )
        token_count = len(frontend.token_ids)
        if token_count > self._capacity.max_tokens:
            raise SynthesisInputTooLongError(
                amount=token_count,
                maximum=self._capacity.max_tokens,
                unit="model_tokens",
            )
        synthesis_config = (
            config.validated() if config is not None else SynthesisConfig(speed=speed).validated()
        )
        style = self.voice_bank.style_for(voice, text_length=len(text))
        effective_speed = self.voice_bank.effective_speed(voice, synthesis_config.speed)

        try:
            result = self.runtime.infer(
                frontend.token_ids,
                style=style,
                speed=effective_speed,
            )
        except KittenSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Kitten model inference failed") from exc
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        calibration_key = self.calibration_key(voice)
        audio, application = apply_voice_level_calibration(
            audio,
            synthesis_config.voice_level,
            calibration_key,
        )
        self._last_voice_level_application = application
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
                "pronunciation_trace": [asdict(item) for item in frontend.pronunciation_trace],
                "pronunciation_diagnostics": [asdict(item) for item in frontend.diagnostics],
                "g2p_identity": self._g2p_configuration_identity(),
                "internal_voice": self.voice_bank.resolve(voice),
                "model_id": self.model_id,
                "voice_level": {
                    "mode": application.mode,
                    "applied": application.applied,
                    "gain_db": application.gain_db,
                    "source": application.source,
                    "calibration_key": str(application.key) if application.key else None,
                    "reason": application.reason,
                    "catalog_revision": application.catalog_revision,
                },
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
            "g2p_identity": self._g2p_configuration_identity(),
        }

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.runtime, "close", None)
        try:
            if callable(close):
                close()
        finally:
            try:
                if self._owns_g2p:
                    self.g2p.close()
            finally:
                self._closed = True

    def __enter__(self) -> KittenVoice:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
