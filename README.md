# kittensynth

KittenTTS synthesis layer built on **kitteng2p + OnnxVoice**.

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

A future/updated OnnxVoice release must register:

```text
system = kitten
```

and expose:

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
git tag v0.1.0
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

Real synthesis remains blocked until OnnxVoice contains the Kitten adapter/catalog parser.
