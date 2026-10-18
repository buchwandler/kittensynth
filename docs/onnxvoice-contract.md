# OnnxVoice Kitten contract required by kittensynth

## Dependency and install contract

KittenSynth depends on OnnxVoice `>=0.2.10,<0.3`. The base package remains provider-neutral: it does not install ONNX Runtime. Optional dependencies are exposed as follows:

- `kittensynth[cpu]` installs OnnxVoice's CPU runtime and Hugging Face transport.
- `kittensynth[gpu]` installs OnnxVoice's GPU runtime and Hugging Face transport.
- `kittensynth[huggingface]` installs the transport without choosing an inference provider.
- `kittensynth[bundled-g2p]` installs the bundled G2P runtime.

The `from_pretrained()` Kitten catalog uses Hugging Face-backed artifacts, so CPU/GPU extras support managed downloads from a cold cache. Local-model users may keep the base package and select their provider arrangement separately; managed downloads use OnnxVoice's verified store and integrity checks.

Register `system="kitten"`.

The managed installation should provide:

```text
artifact("model")
artifact("voices")
sample_rate = 24000
metadata.voice_aliases
metadata.speed_priors
metadata.runtime.profile = ONNX2
```

The runtime adapter surface used by this package is:

```python
InferenceResult infer(
    token_ids: Sequence[int],
    *,
    style: numpy.ndarray,
    speed: float = 1.0,
)
```

OnnxVoice should convert `token_ids` to the model's expected `[1, N]` integer tensor, normalize
`speed` to the expected float tensor, validate `style`, run the graph, trim the upstream v0.8 tail,
and return mono finite float32 audio.

## Public discovery contract

KittenSynth lists the Kitten catalog with `OnnxVoice.list("kitten", language=..., refresh=...)` and reads voice descriptions with `list_voices`. The `offline`, `cache_dir`, and optional catalog source are applied when the manager is created. Listing does not call `install` or `open` and never downloads or initializes a model artifact.

Public `DiscoveredModel` records retain catalog ID/name/version/language/quality/sample rate/aliases, catalog metadata, and a stable `source_revision` both as a field and in metadata. Public voices are user-facing aliases from catalog voice records or `metadata.voice_aliases`. Explicit voice metadata is used as supplied; when aliases are the only voice information, demographics remain unknown and English (`en`) is used as the language/locale default. Default selection is explicit catalog default, then `Jasper`, then the first public alias, else `None`.

Global `runtime_identity()` describes installed software versions and request API version only. Catalog/model revision data remains associated with each discovered model so global identity does not change with the selected catalog item.

## Kitten 0.8 model-input capacity

The supported Kitten 0.8 ONNX artifacts have a 512-row positional-embedding table. This is the hard maximum for the complete `input_ids` sequence sent to OnnxVoice. KittenG2P frames IDs as `(0, *ids, 10, 0)`, and those three frame IDs count toward the maximum.

Kittensynth's centralized capacity policy supplies `DiscoveredModel.max_tokens`, `measure_prepared().max_tokens`, synthesis preflight, and the public `SynthesisInputTooLongError`. Discovery metadata and measurements carry the same `capacity_policy_revision`, derived from the capacity policy, frame IDs, model revision, and installed frontend/runtime versions. Measurement uses the exact framed IDs passed to inference and does not initialize acoustic inference. Over-limit synthesis fails before the runtime call without truncation or splitting.
