from __future__ import annotations

import argparse

from .voice import KittenVoice


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize one prepared KittenTTS request")
    parser.add_argument("text")
    parser.add_argument("--model", default="nano-0.8-int8")
    parser.add_argument("--voice", default="Jasper")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("-o", "--output", default="kitten.wav")
    parser.add_argument("--cache-dir")
    parser.add_argument("--catalog-url")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    with KittenVoice.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        catalog_url=args.catalog_url,
        offline=args.offline,
    ) as model:
        result = model.synthesize_prepared(args.text, voice=args.voice, speed=args.speed)
        result.save_wav(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
