# Architecture

## Ownership

| Concern                                     | Owner                            |
| ------------------------------------------- | -------------------------------- |
| semantic written-to-spoken preparation      | caller                           |
| eSpeak execution                            | `kitteng2p` / `espeakng-runtime` |
| Kitten phoneme tokenization                 | `kitteng2p`                      |
| Kitten token IDs + framing                  | `kitteng2p`                      |
| voice alias/style-row selection             | `kittensynth`                    |
| speed-prior application                     | `kittensynth`                    |
| catalog/download/cache/integrity            | `onnxvoice`                      |
| ONNX Runtime session/providers              | `onnxvoice`                      |
| graph ABI + tail trimming                   | `onnxvoice` Kitten adapter       |
| WAV/container output                        | `kittensynth` convenience result |
| static per-model/per-voice level correction | `kittensynth`                    |
| BS.1770 calibration measurement tooling     | `benchmarks` / `audiosig`        |
| final program loudness / mastering          | caller                           |

## Static voice-level correction

Runtime calibration is an opt-in static gain lookup keyed by the exact managed model ID and Kitten internal voice/style ID. It never infers a managed identity for local models, measures loudness during synthesis, or performs dynamic normalization. The audio path is:

```text
runtime inference
 -> finite mono float32 validation
 -> static voice-level gain (when explicitly enabled)
 -> SynthesisResult
 -> WAV conversion/clamp on save
```

`SynthesisResult.save_wav()` clips to the PCM range. The benchmark therefore reviews predicted and measured peaks; clipping during WAV serialization is not a substitute for safe calibration. The default package mode remains off. See [benchmark documentation](https://github.com/buchwandler/kittensynth/blob/main/benchmarks/README.md) for the measurement/promotion workflow and packaged catalog provenance.

## Why G2P is separate

Kitten's character table and eSpeak-to-ID contract are independently testable and reusable. Keeping
them in `kitteng2p` prevents `kittensynth` from importing the native phonemization stack and mirrors
the existing `piperg2p` / `pipersynth` separation.

## Required OnnxVoice adapter

```python
runtime.infer(token_ids, style=style, speed=speed)
```

OnnxVoice owns graph input dtype/rank validation and the upstream v0.8 `[..., :-5000]` tail
handling needed for parity.

## Public engine API and integration identity

`discover_models()` is KittenSynth's catalog-only public model/voice view. It delegates listing to the OnnxVoice manager and never installs or opens model artifacts. Its frozen records expose display metadata, sample rate, public voice aliases, a deterministic default voice, source revision, and catalog metadata. An explicit catalog default wins; `Jasper` is the package fallback when present, followed by the first public alias. Voice demographics are never inferred from a voice name.

The package-root `runtime_identity()` is a deterministic software identity for PCM-affecting stack versions: engine (`kitten`), KittenSynth, kitteng2p, OnnxVoice, and request API version. It does not include catalog/model revisions, cache paths, timestamps, or runtime objects; catalog/model revisions remain attached to `DiscoveredModel` metadata. `request_api_contract()` truthfully identifies `KittenVoice.synthesize_prepared` as the current API and declares unsupported capabilities false. KittenSynth does not depend on Readio.

Public errors preserve failure categories: invalid requests use request-validation errors, catalog failures use `CatalogUnavailableError` (a `CatalogDiscoveryError` subtype), and graph execution failures use `ModelInferenceError`. OnnxVoice contract mismatches remain `OnnxVoiceContractError`.
