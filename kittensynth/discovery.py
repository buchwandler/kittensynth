"""Public Kitten model and voice discovery.

This module owns the Kitten catalog contact surface. Consumers discover models
and voices through :func:`discover_models`; discovery never installs or opens
model artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._onnxvoice import _manager
from .errors import CatalogDiscoveryError, CatalogUnavailableError, KittenSynthError


@dataclass(frozen=True, slots=True)
class DescribedVoice:
    """Descriptive metadata for one public voice of a Kitten model."""

    id: str
    gender: str = "unknown"
    language: str = "en"
    locale: str = "en"
    language_label: str = "English"
    languages: tuple[str, ...] = ("en",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "languages", tuple(self.languages))


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One discoverable Kitten model with its voices and catalog metadata."""

    id: str
    display_name: str
    version: str | None
    language: str
    quality: str | None
    sample_rate: int
    aliases: tuple[str, ...]
    voices: tuple[DescribedVoice, ...]
    default_voice: str | None
    source_revision: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "voices", tuple(self.voices))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def voice_ids(self) -> tuple[str, ...]:
        """Return the public voice identifiers in catalog order."""
        return tuple(voice.id for voice in self.voices)


def _metadata_value(metadata: Any, key: str, default: str) -> str:
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
    else:
        value = getattr(metadata, key, None)
    return value if isinstance(value, str) and value else default


def _string_values(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,) if values else ()
    if not isinstance(values, Sequence):
        return ()
    return tuple(value for value in values if isinstance(value, str) and value)


def _described_voice(record: Any) -> DescribedVoice:
    metadata = getattr(record, "metadata", None)
    language = _metadata_value(metadata, "language", "en")
    languages = _string_values(getattr(record, "languages", ())) or (language,)
    voice_id = getattr(record, "voice_id", None) or getattr(record, "id", "")
    return DescribedVoice(
        id=str(voice_id),
        gender=_metadata_value(metadata, "gender", "unknown"),
        language=language,
        locale=_metadata_value(metadata, "locale", "en"),
        language_label=_metadata_value(metadata, "language_label", "English"),
        languages=languages,
    )


def _voice_ids(item: Any, metadata: Mapping[str, Any], records: Sequence[Any]) -> tuple[str, ...]:
    aliases = metadata.get("voice_aliases")
    from_aliases = (
        tuple(key for key in aliases if isinstance(key, str) and key)
        if isinstance(aliases, Mapping)
        else ()
    )
    declared = _string_values(getattr(item, "voices", ()))
    recorded = tuple(
        voice_id
        for voice_id in (getattr(record, "voice_id", None) for record in records)
        if isinstance(voice_id, str) and voice_id
    )
    return tuple(dict.fromkeys((*from_aliases, *declared, *recorded)))


def _model_from_item(item: Any, records: tuple[Any, ...]) -> DiscoveredModel:
    raw_metadata = getattr(item, "metadata", None)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    item_id = str(getattr(item, "id", ""))
    item_system = getattr(item, "system", "kitten")
    matching_records = tuple(
        record
        for record in records
        if getattr(getattr(record, "catalog_item", None), "id", None) == item_id
        and getattr(getattr(record, "catalog_item", None), "system", item_system) == item_system
    )
    voices_by_id = {
        voice.id: voice for record in matching_records if (voice := _described_voice(record)).id
    }
    voice_ids = _voice_ids(item, metadata, matching_records)
    voices = tuple(
        voices_by_id.get(voice_id, DescribedVoice(id=voice_id)) for voice_id in voice_ids
    )

    raw_sample_rate = getattr(item, "sample_rate", None)
    sample_rate = (
        raw_sample_rate
        if isinstance(raw_sample_rate, int)
        and not isinstance(raw_sample_rate, bool)
        and raw_sample_rate > 0
        else metadata.get("sample_rate")
    )
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        sample_rate = 24000

    revision = metadata.get("source_revision") or getattr(item, "source_revision", None)
    if not isinstance(revision, str) or not revision:
        revision = None
    if revision is not None:
        metadata["source_revision"] = revision

    version = metadata.get("version")
    language = metadata.get("language")
    quality = metadata.get("quality")
    raw_aliases = getattr(item, "aliases", ())
    aliases = _string_values(raw_aliases)
    explicit_default = getattr(item, "default_voice", None) or metadata.get("default_voice")
    default_voice: str | None
    if isinstance(explicit_default, str) and explicit_default:
        default_voice = explicit_default
    elif "Jasper" in voice_ids:
        default_voice = "Jasper"
    else:
        default_voice = voice_ids[0] if voice_ids else None

    return DiscoveredModel(
        id=item_id,
        display_name=_metadata_value(metadata, "name", item_id),
        version=version if isinstance(version, str) else None,
        language=language if isinstance(language, str) and language else "en",
        quality=quality if isinstance(quality, str) else None,
        sample_rate=sample_rate,
        aliases=aliases,
        voices=voices,
        default_voice=default_voice,
        source_revision=revision,
        metadata=metadata,
    )


def discover_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[DiscoveredModel, ...]:
    """Discover Kitten models and voices from the managed catalog.

    This operation is catalog-only: it neither installs nor opens any model
    artifacts. The OnnxVoice manager owns the catalog, cache, and offline policy.
    """
    try:
        manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
        items = manager.list("kitten", language=language, refresh=refresh)
        records = tuple(manager.list_voices("kitten", language=language, refresh=False))
    except CatalogUnavailableError:
        raise
    except CatalogDiscoveryError as exc:
        raise CatalogUnavailableError(str(exc)) from exc
    except KittenSynthError:
        raise
    except Exception as exc:
        raise CatalogUnavailableError(f"Kitten model catalog is unavailable: {exc}") from exc
    return tuple(_model_from_item(item, records) for item in items if item.system == "kitten")


__all__ = [
    "CatalogDiscoveryError",
    "DescribedVoice",
    "DiscoveredModel",
    "discover_models",
]
