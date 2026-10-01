# Architecture

## Ownership

| Concern                                | Owner                            |
| -------------------------------------- | -------------------------------- |
| semantic written-to-spoken preparation | caller                           |
| eSpeak execution                       | `kitteng2p` / `espeakng-runtime` |
| Kitten phoneme tokenization            | `kitteng2p`                      |
| Kitten token IDs + framing             | `kitteng2p`                      |
| voice alias/style-row selection        | `kittensynth`                    |
| speed-prior application                | `kittensynth`                    |
| catalog/download/cache/integrity       | `onnxvoice`                      |
| ONNX Runtime session/providers         | `onnxvoice`                      |
| graph ABI + tail trimming              | `onnxvoice` Kitten adapter       |
| WAV/container output                   | `kittensynth` convenience result |

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
