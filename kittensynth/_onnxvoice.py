from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import OnnxVoiceContractError, UnsupportedModelError
from .voice_bank import DEFAULT_VOICE_ALIASES

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/buchwandler/kitten-onnx-bundles/main/catalog/models.json"
)


@dataclass(frozen=True, slots=True)
class ResolvedKittenModel:
    ref: str
    model_id: str
    model_path: Path
    voices_path: Path
    sample_rate: int
    metadata: dict[str, Any]
    installation: Any


def _module() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise OnnxVoiceContractError("OnnxVoice is required for Kitten synthesis") from exc
    return onnxvoice


def normalize_ref(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("model reference cannot be empty")
    return value if ":" in value else f"kitten:{value}"


def _manager(
    *,
    cache_dir: str | Path | None,
    offline: bool,
    catalog_url: str | None,
) -> Any:
    module = _module()
    source = catalog_url or os.getenv("KITTENSYNTH_CATALOG_URL") or DEFAULT_CATALOG_URL
    try:
        return module.OnnxVoice(
            cache_dir=cache_dir,
            catalog_sources={"kitten": source},
            offline=offline,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed OnnxVoice does not expose the expected manager constructor"
        ) from exc


def installation_to_model(installation: Any) -> ResolvedKittenModel:
    if str(getattr(installation, "system", "")) != "kitten":
        raise UnsupportedModelError("expected an OnnxVoice Kitten installation")
    try:
        model_path = Path(installation.artifact("model").path)
        voices_path = Path(installation.artifact("voices").path)
    except (KeyError, AttributeError) as exc:
        raise UnsupportedModelError(
            "Kitten installation must contain model and voices artifacts"
        ) from exc

    metadata = dict(getattr(installation, "metadata", {}) or {})
    metadata.setdefault("voice_aliases", dict(DEFAULT_VOICE_ALIASES))
    metadata.setdefault("speed_priors", {})
    sample_rate = int(
        getattr(installation, "sample_rate", None) or metadata.get("sample_rate") or 24000
    )
    return ResolvedKittenModel(
        ref=str(getattr(installation, "ref", f"kitten:{getattr(installation, 'id', 'unknown')}")),
        model_id=str(getattr(installation, "id", "unknown")),
        model_path=model_path,
        voices_path=voices_path,
        sample_rate=sample_rate,
        metadata=metadata,
        installation=installation,
    )


def install_pretrained_model(
    ref: str,
    *,
    quality: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh_catalog: bool = False,
    force_download: bool = False,
    catalog_url: str | None = None,
    progress: Any | None = None,
) -> ResolvedKittenModel:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    installation = manager.install(
        normalize_ref(ref),
        quality=quality,
        refresh=refresh_catalog,
        force=force_download,
        progress=progress,
    )
    return installation_to_model(installation)


def open_installed_model(
    resolved: ResolvedKittenModel,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    catalog_url: str | None = None,
) -> Any:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        return manager.open(
            resolved.installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        if "kitten" in str(exc).lower() or "unsupported system" in str(exc).lower():
            raise OnnxVoiceContractError(
                "OnnxVoice must register a 'kitten' adapter before kittensynth can infer"
            ) from exc
        raise


def open_local_model(
    *,
    model_path: str | Path,
    voices_path: str | Path,
    metadata: Mapping[str, Any],
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    module = _module()
    try:
        return module.OnnxVoice.open_local(
            system="kitten",
            model=model_path,
            voices=voices_path,
            metadata=dict(metadata),
            sample_rate=int(metadata.get("sample_rate") or 24000),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        if "kitten" in str(exc).lower() or "unsupported system" in str(exc).lower():
            raise OnnxVoiceContractError(
                "OnnxVoice must register a 'kitten' adapter before local inference works"
            ) from exc
        raise


def load_local_config(path: str | Path | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "sample_rate": 24000,
        "voice_aliases": dict(DEFAULT_VOICE_ALIASES),
        "speed_priors": {},
        "runtime": {"profile": "ONNX2"},
    }
    if path is None:
        return metadata

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise UnsupportedModelError("Kitten config.json must contain an object")
    profile = raw.get("type")
    if profile not in {"ONNX1", "ONNX2", None}:
        raise UnsupportedModelError(f"unsupported Kitten runtime profile: {profile!r}")
    if isinstance(raw.get("voice_aliases"), dict):
        metadata["voice_aliases"] = {
            str(key): str(value) for key, value in raw["voice_aliases"].items()
        }
    if isinstance(raw.get("speed_priors"), dict):
        metadata["speed_priors"] = {
            str(key): float(value) for key, value in raw["speed_priors"].items()
        }
    metadata["runtime"] = {"profile": profile or "ONNX2"}
    metadata["upstream_config"] = raw
    return metadata
