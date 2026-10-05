# kittensynth

KittenTTS synthesis layer built on **kitteng2p + OnnxVoice**.

## Installation

A base `pip install kittensynth` is provider-neutral and does not install ONNX Runtime. For CPU inference, install the CPU provider extra:

```bash
python -m pip install "kittensynth[cpu]"
```

For GPU inference or the bundled G2P runtime, use:

```bash
python -m pip install "kittensynth[gpu]"
python -m pip install "kittensynth[cpu,bundled-g2p]"
```

Managed examples require an ONNX Runtime provider and a working G2P runtime. The CPU plus bundled G2P command above is the copy/paste setup for the examples.

## Documentation

- [Architecture](docs/architecture.md)
- [OnnxVoice contract](docs/onnxvoice-contract.md)
- [Calibration and benchmark methodology](benchmarks/README.md)

This MVP follows the same repository/package shape as PiperSynth/PiperG2P:

```text
pyproject.toml
kittensynth/
tests/
docs/
```

There is **no `src/` layer** and no hard-coded package version.

## Architecture

```text
prepared speakable English
        |
        v
kitteng2p
  - eSpeak
  - Kitten phoneme tokenization
  - Kitten v0.8 token IDs
        |
        v
kittensynth
  - voice alias/style row
  - speed-prior policy
        |
        v
onnxvoice
  - catalogs/install/cache/integrity
  - ORT providers/sessions
  - Kitten graph ABI
        |
        v
float32 waveform
```

`kittensynth` has **no direct eSpeak or phonemizer dependency**. That is now entirely owned by
`kitteng2p`.

## Required OnnxVoice contract

KittenSynth requires OnnxVoice 0.2.x. OnnxVoice 0.2.0 and later in that supported range register the built-in Kitten adapter and expose:

```text
system = kitten
```

```python
runtime.infer(token_ids, style=style, speed=effective_speed)
```

The installation contains at least:

```text
role=model
role=voices
```

with 24 kHz metadata and Kitten voice aliases/speed priors.

## Managed model

```python
from kittensynth import KittenVoice

with KittenVoice.from_pretrained("nano-0.8-int8") as model:
    result = model.synthesize_prepared(
        "Hello from KittenSynth.",
        voice="Jasper",
    )
    result.save_wav("hello.wav")
```

## Public discovery and Readio adapter contract

`discover_models()` exposes the OnnxVoice Kitten catalog as immutable `DiscoveredModel` and `DescribedVoice` records. It lists catalog data only: it does not install or open model artifacts. `language`, `offline`, `refresh`, `cache_dir`, and `catalog_url` are forwarded to the managed catalog layer. `source_revision` is available on each model and in its metadata; model-specific revisions belong with the model, not the global engine identity.

```python
from kittensynth import discover_models, runtime_identity

models = discover_models(language="en", offline=True)
for model in models:
    print(model.id, model.display_name, model.voice_ids, model.default_voice)

print(runtime_identity())
```

A catalog default voice takes precedence; otherwise KittenSynth selects `Jasper` when listed, then the first public alias. Catalog-provided voice metadata is preserved. When a catalog provides only `metadata.voice_aliases`, voices are described with unknown gender and English language defaults; gender is never guessed from an alias.

`runtime_identity()` returns deterministic engine, KittenSynth, kitteng2p, OnnxVoice, and request-API versions. It contains no paths or machine-local state. `request_api_contract()` declares `KittenVoice.synthesize_prepared` as the current entrypoint: prepared-text boundaries remain caller-owned, and linguistic tokens, pronunciation overrides, whole-request phonemes, speakers, word timings, and request-level voice controls are not supported. The package exports typed errors such as `EmptyTextError`, `InvalidVoiceError`, `OnnxVoiceContractError`, `CatalogUnavailableError`, and `ModelInferenceError`; no Readio dependency is required.

## Local model

```python
from kittensynth import KittenVoice

with KittenVoice.from_local(
    model_path="kitten_tts_nano_v0_8.onnx",
    voices_path="voices.npz",
    config_path="config.json",
) as model:
    result = model.synthesize_prepared("Local synthesis.", voice="Bella")
    result.save_wav("local.wav")
```

## Examples

Run the prepared-speech examples with a managed Kitten model:

```bash
python examples/basic.py
python examples/all_voices.py
python examples/run_all.py --list
```

The first managed run may download model assets. Generated WAVs and the all-voices manifest are written under `example-artefacts/` and intentionally gitignored. Use `python examples/run_all.py` to execute each example in an isolated output directory and validate its WAV files.

## Static voice-level calibration

Calibration is an optional deterministic static gain correction. The package default remains `off` for backward compatibility; opt in explicitly:

```python
from kittensynth import KittenVoice, SynthesisConfig, VoiceLevelConfig

config = SynthesisConfig(
    speed=1.0,
    voice_level=VoiceLevelConfig(mode="calibrated"),
)
with KittenVoice.from_pretrained("nano-0.8-int8") as model:
    result = model.synthesize_prepared("Prepared speech.", voice="Jasper", config=config)
```

### Calibration defaults by interface

| Interface                                     | Default behavior                                          |
| --------------------------------------------- | --------------------------------------------------------- |
| Python `KittenVoice.synthesize_prepared(...)` | Calibration off                                           |
| `examples/basic.py`                           | Calibrated                                                |
| `examples/all_voices.py`                      | Calibrated; `--raw` disables it                           |
| `kittensynth` CLI                             | Calibration off; use `--voice-level calibrated` to opt in |

The CLI also accepts `--gain-db FLOAT` for an explicit static gain override. The API and CLI defaults remain off for backward compatibility.

### Speed precedence

When `config` is provided, `config.speed` is authoritative and the separate `speed=` argument is ignored:

```python
model.synthesize_prepared(
    "Prepared speech.",
    speed=1.2,  # ignored because config is supplied
    config=SynthesisConfig(speed=0.9),
)
```

Catalog lookup uses the exact managed model ID and internal voice/style ID, not the friendly alias. The packaged measured catalog covers all eight internal voices for `micro-0.8`, `mini-0.8`, `nano-0.8-int8`, and `nano-0.8-fp32`; local models do not inherit managed gains. A reviewed explicit `gain_db` override is available for a local model. Calibration is neither request-time loudness measurement nor dynamic normalization/limiting. See [benchmark documentation](benchmarks/README.md) for the reproducible measurement and promotion workflow and catalog provenance.

## Prepared-text boundary

Like PiperG2P/PiperSynth, this MVP expects prepared speakable text. Written-form semantic expansion
(numbers, currencies, dates, URLs, abbreviations) stays outside the engine.

That keeps these packages independent:

```text
semantic preparation -> kitteng2p -> kittensynth -> onnxvoice
```

## Voice selection

The current v0.8 aliases are:

```text
Bella   -> expr-voice-2-f
Jasper  -> expr-voice-2-m
Luna    -> expr-voice-3-f
Bruno   -> expr-voice-3-m
Rosie   -> expr-voice-4-f
Hugo    -> expr-voice-4-m
Kiki    -> expr-voice-5-f
Leo     -> expr-voice-5-m
```

The style row is selected exactly as upstream v0.8:

```python
min(len(text), style_rows - 1)
```

## Dynamic versioning

Both `kitteng2p` and `kittensynth` use the same Git-tag-driven `setuptools-scm` pattern:

```bash
git tag vX.Y.Z
python -m build
```

The source-ZIP fallback is `0.1.dev0`; installed `__version__` comes from distribution metadata.

## Development

For sibling checkout development:

```bash
python -m pip install -e ../kitteng2p
python -m pip install -e ".[dev]"
python -m pytest
```

Managed synthesis is provided by the Kitten adapter in OnnxVoice; see the examples and static calibration sections above for runnable usage.
