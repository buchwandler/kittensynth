"""Authoritative model-input capacity for the supported Kitten 0.8 contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from kitteng2p import FRAME_PREFIX_ID, FRAME_SUFFIX_IDS

from .identity import runtime_identity

MAX_MODEL_INPUT_TOKENS = 512
_CAPACITY_POLICY_VERSION = "kitten-model-input-v1"


@dataclass(frozen=True, slots=True)
class ModelCapacity:
    max_tokens: int
    policy_revision: str


def capacity_for_model(
    model_id: str | None,
    source_revision: str | None,
) -> ModelCapacity:
    identity = runtime_identity()
    revision_inputs: dict[str, Any] = {
        "policy": _CAPACITY_POLICY_VERSION,
        "max_tokens": MAX_MODEL_INPUT_TOKENS,
        "frame_prefix_id": FRAME_PREFIX_ID,
        "frame_suffix_ids": FRAME_SUFFIX_IDS,
        "model_id": model_id or "local",
        "model_revision": source_revision or "unknown",
        "frontend_revision": identity["g2p_revision"] or "unknown",
        "runtime_revision": identity["runtime_revision"] or "unknown",
    }
    serialized = json.dumps(revision_inputs, sort_keys=True, separators=(",", ":"))
    revision = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ModelCapacity(
        max_tokens=MAX_MODEL_INPUT_TOKENS,
        policy_revision=f"sha256:{revision}",
    )


__all__ = ["MAX_MODEL_INPUT_TOKENS", "ModelCapacity", "capacity_for_model"]
