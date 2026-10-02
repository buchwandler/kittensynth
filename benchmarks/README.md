# KittenSynth static voice calibration

The benchmark measures real, uncalibrated Kitten synthesis. It opens the selected managed model once, exercises the prepared short/medium/long stimulus set for every selected voice and configured repeat, and records BS.1770 integrated loudness, model/voice identity, exact stimuli, repeat measurements, failures, tool versions, and peak data. Runtime synthesis never measures loudness.

```bash
python benchmarks/voice_level_benchmark.py --model nano-0.8-int8 --list-only
python benchmarks/voice_level_benchmark.py --model nano-0.8-int8
python benchmarks/promote_voice_calibration.py
```

For multiple reports, pass each report path as a positional argument. Each report is validated independently, catalog metadata must match, and the voice keys must be disjoint. The no-argument default and single-report invocation remain supported:

```bash
python benchmarks/promote_voice_calibration.py \
  benchmarks/output/voice_level_calibration/micro-0.8.json \
  benchmarks/output/voice_level_calibration/mini-0.8.json \
  --output benchmarks/output/voice_level_calibration/candidate_catalog.json
```

The default measurement report is `benchmarks/output/voice_level_calibration/measurements.json`; the default candidate is `benchmarks/output/voice_level_calibration/candidate_catalog.json`. Both are generated artifacts and are gitignored. Use `--offline`, `--cache-dir`, `--quality`, `--voice`, `--refresh-catalog`, `--output`, `--stimuli`, and `--policy` to control measurement.

Review complete coverage, repeat variability, policy-bound gains, and predicted post-gain peaks before promotion. Incomplete coverage requires `--allow-partial`; high-variability identities are omitted unless explicitly included; predicted clipping risk and gains at policy bounds require explicit review flags. Promotion writes only a separate candidate and refuses the packaged production catalog path. Copy reviewed, measured candidate data to `kittensynth/data/voice_level_calibration.json` deliberately; never fabricate, infer, or reuse gains across model IDs.

## Packaged catalog measurement provenance

The packaged catalog contains the exact managed model IDs `micro-0.8`, `mini-0.8`, `nano-0.8-int8`, and `nano-0.8-fp32`, with all eight internal voices for each model (32 calibration identities total). Each model report used `kittensynth-prepared-speech-v1` with the prepared short/medium/long stimuli and three repeats per stimulus, yielding 72 measurements per model and 288 promoted measurements overall. Every promoted identity had complete coverage, `eligible` status, and uncalibrated synthesis (`calibration_mode="off"`).

A benchmark attempt for `nano-0.8` is intentionally not promoted: the managed catalog lookup failed with `AssetNotFoundError: Unknown catalog item: kitten:nano-0.8`, producing zero measured identities. Calibration keys are exact model IDs, so gains from another Nano variant must not be reused for that missing identity.

The four promoted reports used the same policy (`reference_lufs=-24.0`, gain range `[-12.0, 8.0]`, maximum repeat MAD `0.75 LU`) and the same tool versions: KittenSynth `0.1.0`, kitteng2p `0.1.dev1+g3caf05c9b.d19800101`, OnnxVoice `0.2.0`, and audiosig `0.1.6`. Across all 32 identities, gains range from `-5.4425721305 dB` to `-0.9478686520 dB`, the maximum repeat MAD is `0.0570080321 LU`, the maximum cross-stimulus spread is `2.5057701869 LU`, and the largest predicted post-gain peak is `0.8296009469`. No promoted gain hit a policy bound and no identity required peak-safety review.

Report SHA-256 values for all supplied reports. The `nano-0.8` attempt failed and was not promoted:

| Model/report                      | Report SHA-256                                                     |
| --------------------------------- | ------------------------------------------------------------------ |
| `micro-0.8`                       | `03cf3b99691d56cd5b41217e05f17990309ddb8b506823e48cb6a65847f5148d` |
| `mini-0.8`                        | `dad4211f2956e8346fd5e1219ff9b94ce851e28958e1c59487aeb0d774490b68` |
| `nano-0.8-int8`                   | `d41f065367f18084b8b0c10067e62894a544919a8a95192f285f2f5d46734cce` |
| `nano-0.8-fp32`                   | `6b7875e95be4b16d483f7a04ad91f944e14de76b4640d810011af87cd35e94ba` |
| `nano-0.8` (failed, not promoted) | `07baf43ae76550f838faee774db751c61c6cb9da18fbc5dbbef961c91665800c` |

The merged packaged catalog SHA-256 is `ed84099411f6cf7c9607a67b496b7794f85ee8a3e693cf2df150ce94e9ae95f7`.

Before release, run a second real synthesis pass with `VoiceLevelConfig(mode="calibrated")` for all 32 identities and verify the observed output loudness and peak headroom. The catalog is static: runtime synthesis applies only the stored gain for the exact managed model/internal-voice key and does not measure loudness dynamically.
