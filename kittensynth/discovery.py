"""Public Kitten model and voice discovery.

This module owns the Kitten catalog contact surface. Consumers discover models
and voices through :func:`discover_models` and never construct the underlying
asset runtime themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

from ._onnxvoice import _manager
from .errors import CatalogDiscoveryError


@dataclass(frozen=True, slots=True)
class DescribedVoice:
    """Descriptive metadata for one voice of a Kitten model."""

    id: str
    gender: str = "unknown"
    language: str = "unknown"
    locale: str = "unknown"
    language_label: str = "unknown"
    languages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One discoverable Kitten model with its voices and revisions."""

    id: str
    display_name: str
    version: str | None
    language: str | None
    quality: str | None
    sample_rate: int | None
    aliases: tuple[str, ...]
    voices: tuple[DescribedVoice, ...]
    default_voice: str | None
    source_revision: str | None
    runtime_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def voice_ids(self) -> tuple[str, ...]:
        return tuple(voice.id for voice in self.voices)


def _distribution_version_safe(distribution: str) -> str | None:
    try:
        return _distribution_version(distribution)
    except PackageNotFoundError:
        return None


def _metadata_value(metadata: Any, key: str, default: str) -> str:
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
    else:
        value = getattr(metadata, key, None)
    return value if isinstance(value, str) and value else default


def _described_voice(record: Any) -> DescribedVoice:
    metadata = getattr(record, "metadata", None)
    languages = tuple(
        str(code) for code in (getattr(record, "languages", ()) or ()) if isinstance(code, str)
    )
    return DescribedVoice(
        id=str(getattr(record, "voice_id", "")),
        gender=_metadata_value(metadata, "gender", "unknown"),
        language=_metadata_value(metadata, "language", "unknown"),
        locale=_metadata_value(metadata, "locale", "unknown"),
        language_label=_metadata_value(metadata, "language_label", "unknown"),
        languages=languages,
    )


def _model_from_item(item: Any, records: tuple[Any, ...]) -> DiscoveredModel:
    metadata = dict(item.metadata) if isinstance(item.metadata, Mapping) else {}
    sample_rate = getattr(item, "sample_rate", None)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        sample_rate = None
    revision = metadata.get("source_revision")
    version = metadata.get("version")
    language = metadata.get("language")
    quality = metadata.get("quality")
    voices = tuple(
        _described_voice(record)
        for record in records
        if getattr(getattr(record, "catalog_item", None), "id", None) == item.id
    )
    if not voices:
        voices = tuple(
            DescribedVoice(id=str(voice_id))
            for voice_id in getattr(item, "voices", ())
            if isinstance(voice_id, str) and voice_id
        )
    return DiscoveredModel(
        id=str(item.id),
        display_name=str(metadata.get("name") or item.id),
        version=version if isinstance(version, str) else None,
        language=language if isinstance(language, str) else None,
        quality=quality if isinstance(quality, str) else None,
        sample_rate=sample_rate,
        aliases=tuple(str(alias) for alias in (getattr(item, "aliases", ()) or ())),
        voices=voices,
        default_voice=getattr(item, "default_voice", None),
        source_revision=revision if isinstance(revision, str) else None,
        metadata=metadata,
    )


def discover_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: Any | None = None,
    catalog_url: str | None = None,
) -> tuple[DiscoveredModel, ...]:
    """Discover Kitten models and their voices from the managed catalog.

    Discovery is read-only with respect to model artifacts: no model is
    downloaded or opened.
    """
    try:
        manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
        items = manager.list("kitten", language=language, refresh=refresh)
        records = tuple(manager.list_voices("kitten", language=language, refresh=False))
    except CatalogDiscoveryError:
        raise
    except Exception as exc:
        raise CatalogDiscoveryError(f"Kitten model catalog is unavailable: {exc}") from exc
    return tuple(_model_from_item(item, records) for item in items if item.system == "kitten")


def runtime_identity(model: DiscoveredModel | None = None) -> dict[str, str | None]:
    """Return an opaque engine runtime identity for cache and provenance keys.

    Callers persist the mapping without interpreting the package names behind it.
    """
    identity: dict[str, str | None] = {
        "engine_version": _distribution_version_safe("kittensynth"),
        "runtime_revision": _distribution_version_safe("onnxvoice"),
        "g2p_revision": _distribution_version_safe("kitteng2p"),
        "catalog_revision": None,
        "model_revision": None,
    }
    if model is not None:
        identity["catalog_revision"] = model.source_revision
        identity["model_revision"] = model.version or model.source_revision
    return identity


__all__ = [
    "CatalogDiscoveryError",
    "DescribedVoice",
    "DiscoveredModel",
    "discover_models",
    "runtime_identity",
]
