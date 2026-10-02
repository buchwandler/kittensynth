# KittenSynth static voice calibration

The benchmark measures real, uncalibrated Kitten synthesis. It opens the selected managed model once, exercises the prepared short/medium/long stimulus set for every selected voice and configured repeat, and records BS.1770 integrated loudness, model/voice identity, exact stimuli, repeat measurements, failures, tool versions, and peak data. Runtime synthesis never measures loudness.

```bash
python benchmarks/voice_level_benchmark.py --model nano-0.8-int8 --list-only
python benchmarks/voice_level_benchmark.py --model nano-0.8-int8
python benchmarks/promote_voice_calibration.py
```

The default measurement report is `benchmarks/output/voice_level_calibration/measurements.json`; the default candidate is `benchmarks/output/voice_level_calibration/candidate_catalog.json`. Both are generated artifacts and are gitignored. Use `--offline`, `--cache-dir`, `--quality`, `--voice`, `--refresh-catalog`, `--output`, `--stimuli`, and `--policy` to control measurement.

Review complete coverage, repeat variability, policy-bound gains, and predicted post-gain peaks before promotion. Incomplete coverage requires `--allow-partial`; high-variability identities are omitted unless explicitly included; predicted clipping risk and gains at policy bounds require explicit review flags. Promotion writes only a separate candidate and refuses the packaged production catalog path. Copy reviewed, measured candidate data to `kittensynth/data/voice_level_calibration.json` deliberately; never fabricate, infer, or reuse gains across model IDs.

## Packaged catalog measurement provenance

The checked-in catalog covers only the exact managed identity `nano-0.8-int8` and its eight internal voice IDs. It was promoted from the complete, uncalibrated report for `kittensynth-prepared-speech-v1` (short/medium/long, three repeats each): 72/72 measurements, 8/8 identities, zero failures. The report records requested speed 1.0, effective speeds 0.8/0.9, and 24 kHz output. Tool versions were KittenSynth 0.1.0, kitteng2p 0.1.1, onnxvoice 0.2.0, and audiosig 0.1.6.

Review results: maximum repeat MAD was 0.0592 LU against the 0.75 LU policy threshold; maximum cross-stimulus spread was 2.4197 LU and is retained as a diagnostic for the expected text-length/style differences. All eight gains were uncapped and negative; the largest predicted post-gain peak was 0.8377, so there was no positive-gain clipping risk. The raw report SHA-256 is `d7c57be4e9554dd72616c26dac155db1629a941e09b5d7c2537b3aecd0966f75`; the reviewed candidate and packaged catalog SHA-256 are both `343ac5fc36ada3d0831494dea30efc9bbe07d9cebfa97e74b4a5bdded31429a4`. The generated report and candidate remain gitignored.

A second real pass synthesized and measured every voice/stimulus/repeat with runtime calibrated mode. All 72 applications used the matching catalog key, stayed below full scale (maximum observed peak 0.8582), and met the recommended absolute median error tolerance of 0.5 LU:

| Voice  | Calibrated corpus median (LUFS) | Error (LU) |
| ------ | ------------------------------: | ---------: |
| Bella  |                         -23.892 |      0.108 |
| Bruno  |                         -24.010 |      0.010 |
| Hugo   |                         -23.969 |      0.031 |
| Jasper |                         -24.015 |      0.015 |
| Kiki   |                         -23.894 |      0.106 |
| Leo    |                         -24.010 |      0.010 |
| Luna   |                         -24.002 |      0.002 |
| Rosie  |                         -23.957 |      0.043 |
