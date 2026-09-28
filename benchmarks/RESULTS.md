# Benchmarks

Native Apple-Silicon DiariZen vs the upstream PyTorch pipeline.

- **Hardware:** Apple M2 Pro (8P + 4E cores), 32 GB, macOS 27.0 (26A428).
- **Model:** `diarizen-wavlm-base-s80-md`.
- **Fixture:** `EN2002a_30s` — 30 s, 16 kHz mono (AMI, 3 speakers).
- **Method:** median wall-clock, **model loading excluded**, 1 warm-up run.
  Swift = release build, 10 runs; PyTorch = 5 runs. Peak RSS via `/usr/bin/time -l`.
- **Reproduce:** `benchmarks/run.sh`.
- **Accuracy:** the native Swift pipeline reproduces the pyannote (CoreML-segmentation)
  RTTM at **DER 0.495%** on this fixture (`validation/compare_swift_rttm.py`). RTF is
  not traded against accuracy.

## Results

| Backend | seg | embed | cluster | recon | **total** | **RTF** | **peak RSS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| native CoreML — **CPU+GPU** (default) | 0.326 | 1.307 | 0.001 | 0.002 | **1.64 s** | **0.055** | **226 MB** |
| native CoreML — CPU only | 0.616 | 1.814 | 0.001 | 0.002 | 2.44 s | 0.081 | 256 MB |
| PyTorch — CPU (reference) | — | — | — | — | 18.04 s | 0.602 | 7474 MB |

(stage columns are seconds; RTF = processing_time / 30 s, lower is faster.)

**Native CoreML+GPU is ~11× faster than PyTorch-CPU (RTF 0.055 vs 0.602) and uses
~33× less memory (226 MB vs 7.3 GB)** — real-time factor ≈ 18× faster than the audio.

## Notes / honest caveats

- **ANE is excluded, not benchmarked.** Any compute-unit policy that includes the
  Neural Engine (`.all`, `.cpuAndNeuralEngine`) **deadlocks** the first segmentation
  predict on this WavLM+Conformer model (hangs at ~0 % CPU; see
  `Sources/DiariZen/CoreMLBackend.swift`). The shipped default is `.cpuAndGPU`. We do
  **not** claim "runs on the ANE" — profiling and fixing that path is Phase 4/later work.
- **Embedding dominates** (~80 % of the time): 40 WeSpeaker ResNet CoreML predicts +
  the Swift kaldi fbank per 30 s (one per chunk × speaker). Clustering + reconstruction
  are <3 ms combined.
- **Short fixture.** 30 s is enough to compare backends but longer audio would give
  steadier RTF; embedding count grows with chunks × speakers.
- **PyTorch RSS** includes the Python/torch/pyannote runtime — the honest footprint of
  running DiariZen in Python, but not a like-for-like "model only" figure.
- **Not benchmarked:** ONNX Runtime (CPU / CoreML EP) and PyTorch-MPS *full pipelines*.
  Only segmentation was ported to ONNX (the reference backend); the complete-pipeline
  reference here is PyTorch-CPU. Adding those is future work.
- **Energy / per-core CPU** not measured (would need `powermetrics`, root). Deferred.
