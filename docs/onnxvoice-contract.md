# OnnxVoice Kitten contract required by kittensynth

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
