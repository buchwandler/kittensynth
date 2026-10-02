#!/usr/bin/env python3
"""Synthesize one prepared stimulus with every available managed Kitten voice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kittensynth import KittenVoice, SynthesisConfig, VoiceLevelConfig

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

DEFAULT_TEXT = "A prepared speech sample for comparing every Kitten voice."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nano-0.8-int8")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--raw", action="store_true", help="disable static voice calibration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    level_config = VoiceLevelConfig(mode="off" if args.raw else "calibrated")
    config = SynthesisConfig(speed=args.speed, voice_level=level_config)
    manifest: dict[str, object] = {
        "requested_model": args.model,
        "voices": [],
    }
    entries: list[dict[str, object]] = []

    with KittenVoice.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh_catalog=args.refresh_catalog,
    ) as model:
        manifest["resolved_model_id"] = model.model_id
        for alias in model.available_voices:
            internal_voice = model.voice_bank.resolve(alias)
            result = model.synthesize_prepared(args.text, voice=alias, config=config)
            calibration = (result.metadata or {}).get("voice_level", {})
            slug = alias.casefold().replace(" ", "-")
            relative_wav = Path("all_voices") / f"{slug}.wav"
            wav_path = artefact_path(relative_wav)
            result.save_wav(wav_path)
            entries.append(
                {
                    "alias": alias,
                    "internal_voice_id": internal_voice,
                    "requested_model": args.model,
                    "resolved_model_id": model.model_id,
                    "effective_speed": result.speed,
                    "duration_seconds": result.duration_seconds,
                    "sample_rate": result.sample_rate,
                    "calibration_mode": calibration.get("mode", level_config.mode),
                    "calibration_source": calibration.get("source"),
                    "applied_db_gain": calibration.get("gain_db", 0.0),
                    "calibration_key": calibration.get("calibration_key"),
                    "catalog_revision": calibration.get("catalog_revision"),
                    "wav_path": relative_wav.as_posix(),
                }
            )
            print(
                f"{alias}\t{internal_voice}\t{result.duration_seconds:.2f}s\t"
                f"{result.sample_rate} Hz\t{wav_path}"
            )

    manifest["voices"] = entries
    manifest_path = artefact_path("all_voices/manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Synthesized {len(entries)} voices. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
