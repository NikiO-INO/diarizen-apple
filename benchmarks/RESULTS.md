# Benchmarks

Reproducible measurements per backend and Apple Silicon generation. **Do not
publish claims that aren't in this table.** In particular, don't write "runs on
the ANE" — write "CoreML backend optimized for the Apple Neural Engine" unless a
profiler (Instruments → Core ML / ANE) confirms ANE residency.

## Method
- Fixture: `validation/fixtures/<name>.wav` (state length + #speakers).
- Warm-up 1 run, then median of N≥5.
- RTF = audio_seconds / wall_seconds (higher = faster than real time).
- Peak memory: `/usr/bin/time -l` (macOS) max RSS.
- Energy (optional): `powermetrics` or Instruments Energy Log.

## Results (fill in)

| Backend | Model | Chip | RTF | Latency (s) | Peak mem (MB) | CPU % | Energy | Notes |
|---|---|---|---|---|---|---|---|---|
| PyTorch / MPS | base-s80-md | — | | | | | | reference |
| ONNX Runtime CPU | base-s80-md | — | | | | | | reference |
| ONNX Runtime + CoreML EP | base-s80-md | — | | | | | | note graph partitions / CPU fallbacks |
| Native CoreML (.all) | base-s80-md | — | | | | | | production |
| Native CoreML (.cpuAndNeuralEngine) | base-s80-md | — | | | | | | |

Repeat for `large-s80-md-v2` once it passes parity.
