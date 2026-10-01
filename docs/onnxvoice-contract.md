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
