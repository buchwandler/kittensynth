# KittenSynth examples

These examples use already prepared, speakable English text. They do not perform semantic text preparation or written-form expansion. The first managed run may download model assets; subsequent runs reuse the cache.

Managed examples need an ONNX Runtime provider and a working G2P runtime. For a CPU setup with bundled G2P:

```bash
python -m pip install "kittensynth[cpu,bundled-g2p]"
```

For GPU inference, install `kittensynth[gpu]` and a supported G2P runtime. A bare `pip install kittensynth` is provider-neutral and does not install ONNX Runtime.

Generated WAVs and manifests are written below `example-artefacts/` unless `KITTENSYNTH_EXAMPLE_OUTPUT_DIR` points to another output directory. Generated files are intentionally gitignored. `basic.py` and `all_voices.py` both use packaged per-model static gains by default. Pass `--raw` to `all_voices.py` to disable calibration for comparison.

```bash
python examples/basic.py
python examples/all_voices.py
python examples/all_voices.py --raw
python examples/run_all.py --list
python examples/run_all.py
```

`basic.py` accepts `KITTENSYNTH_EXAMPLE_MODEL` and `KITTENSYNTH_EXAMPLE_VOICE`. `all_voices.py` accepts `--model`, `--text`, `--speed`, `--cache-dir`, `--offline`, and `--refresh-catalog`; it opens one managed model and creates one WAV for each available voice plus a JSON manifest.
