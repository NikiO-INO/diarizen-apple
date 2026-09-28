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
| native CoreML — CPU+ANE | 4.026 | 1.089 | 0.001 | 0.002 | 5.13 s | 0.171 | 1181 MB |
| PyTorch — MPS | — | — | — | — | 1.43 s | 0.048 | 1832 MB |
| PyTorch — CPU (reference) | — | — | — | — | 18.04 s | 0.602 | 7474 MB |

(stage columns are seconds; RTF = processing_time / 30 s, lower is faster.)

**Takeaways (honest):**
- vs **PyTorch-CPU**: native CoreML+GPU is **~11× faster** (RTF 0.055 vs 0.602) and uses
  **~33× less memory** (226 MB vs 7.3 GB).
- vs **PyTorch-MPS**: raw speed is a **wash** — MPS is marginally faster (0.048 vs 0.055).
  The native port's win is **memory (~8×: 226 MB vs 1832 MB)** and **deployability**: a
  self-contained Swift binary + CoreML models, no Python / torch / pyannote runtime.
  MPS also silently CPU-fallbacks some ops (`aten::_weight_norm_interface`).
- So the pitch is "**runs natively on macOS, tiny footprint, no Python, competitive
  speed**" — not "fastest in wall-clock".

## Notes / honest caveats

- **ANE: investigated — works via `.cpuAndNeuralEngine`, but not worth it here; `.all`
  deadlocks.** The earlier "ANE deadlock" was two separate things: (1) `.all`
  (CPU+GPU+ANE) genuinely **deadlocks** on the first predict (~0 % CPU) — a GPU+ANE
  partitioning conflict; do not use it. (2) `.cpuAndNeuralEngine` runs fine, but WavLM
  segmentation is **~12× slower on the ANE than the GPU** (0.40 s vs 0.033 s per chunk)
  and uses ~5× the memory — transformer segmentation is not ANE-friendly. The embedding
  ResNet *is* slightly faster on the ANE (1.09 s vs 1.31 s), but not enough to matter.
  So we do **not** claim "runs on the ANE" as a win; `.cpuAndGPU` is the shipped default.
  (First-ever ANE compilation of the segmentation model is also slow — minutes — which
  is what the original 13-minute "hang" turned out to be, before it caches.) ANE is
  *accurate* — end-to-end DER on `.cpuAndNeuralEngine` is 0.505% vs 0.495% on CPU — just
  not fast for this model.
- **Embedding dominates** (~80 % of the time): 40 WeSpeaker ResNet CoreML predicts +
  the Swift kaldi fbank per 30 s (one per chunk × speaker). Clustering + reconstruction
  are <3 ms combined.
- **Short fixture.** 30 s is enough to compare backends but longer audio would give
  steadier RTF; embedding count grows with chunks × speakers.
- **PyTorch RSS** includes the Python/torch/pyannote runtime — the honest footprint of
  running DiariZen in Python, but not a like-for-like "model only" figure.
- **Not benchmarked:** ONNX Runtime (CPU / CoreML EP) *full pipeline* — only
  segmentation was ported to ONNX (the reference backend), and ORT is not a production
  target here, so a full ORT pipeline would duplicate the CoreML work for little value.
- **Energy:** `benchmarks/energy.sh` (needs `sudo` for `powermetrics`) samples system-wide
  CPU/GPU/ANE power while the pipeline loops — indicative, not per-process. Run it to
  confirm ANE power stays ~0 under the default `.cpuAndGPU` policy.
