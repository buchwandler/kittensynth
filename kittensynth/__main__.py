from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import SynthesisConfig
from .voice import KittenVoice
from .voice_level import VoiceLevelConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize one prepared KittenTTS request")
    parser.add_argument("text")
    parser.add_argument("--model", default="nano-0.8-int8")
    parser.add_argument("--voice", default="Jasper")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--voice-level",
        choices=("off", "calibrated"),
        default="off",
        help="select static catalog calibration (default: off)",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        help="apply an explicit static gain override in dB",
    )
    parser.add_argument("-o", "--output", default="kitten.wav")
    parser.add_argument("--cache-dir")
    parser.add_argument("--catalog-url")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = SynthesisConfig(
        speed=args.speed,
        voice_level=VoiceLevelConfig(mode=args.voice_level, gain_db=args.gain_db),
    )
    with KittenVoice.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        catalog_url=args.catalog_url,
        offline=args.offline,
    ) as model:
        result = model.synthesize_prepared(args.text, voice=args.voice, config=config)
        result.save_wav(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
