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

## Accuracy on a full meeting (real ground truth)

Beyond the 30 s parity clip, validated on the **full AMI EN2002a** (35.7 min, SDM,
4 speakers) against the human reference (`pyannote/AMI-diarization-setup`), via
`validation/eval_ami.py` + `pyannote.metrics`:

| | DER vs reference |
|---|---:|
| **Swift port — base-s80-md** | **21.16%** |
| DiariZen base (PyTorch) | 21.10% |
| Swift vs DiariZen base (fidelity) | **1.90%** |
| **Swift port — large-s80-md-v2** | **17.50%** |

Switching to `--models build/coreml-large` (large-s80-md-v2) improves DER to **17.50%**
on the same file — the bigger model's gain (tracks DiariZen's published base→large
improvement). Note large-v2's CoreML segmentation diverges from PyTorch at the tensor
level (~29% argmax, accumulates through its 24 WavLM layers) but is tolerated
downstream; base-s80-md remains the tensor-faithful default (1.90% fidelity).

The base port reproduces DiariZen's accuracy on a real long meeting (0.06 pt from the
PyTorch pipeline vs the reference; 1.90% DER between the two outputs). ~21% is this
single hard meeting's difficulty — DiariZen's published **15.8%** is the AMI-SDM
*corpus average*, and PyTorch scores the same ~21% here. The 35-min file exercised
**N = 2238** clustering embeddings (which forced the O(n²) AHC — see the perf commit).
Runtime: Swift **151 s** (RTF 0.071) vs PyTorch-CPU **3042 s** (~20×).

## Perf notes

- **Embedding batching.** The embedding ResNet is exported with a flexible batch dim
  (`RangeDim`), and the pipeline runs speaker-chunks in batches of 32 (one CoreML call
  per batch). This ~halved the embedding stage (35-min AMI: 220 s → 103 s; total 288 s
  → 151 s) with **no accuracy change** (30-s DER stays 0.495%; AMI DER stays 21.16%).
  The kaldi fbank is also computed once per chunk (shared across its speakers).
- **AHC is O(n²)** (nearest-neighbor cache), not the naive O(n³) — exact centroid
  linkage can't beat O(n²), and clustering is <3% of runtime anyway (4.4 s on the
  N=2238 AMI file). Embedding dominates (~70%).

## Results (speed)

| Backend | seg | embed | cluster | recon | **total** | **RTF** | **peak RSS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| native CoreML — **CPU+GPU** (default) | 0.323 | 0.794 | 0.001 | 0.002 | **1.12 s** | **0.037** | **298 MB** |
| native CoreML — CPU only | 0.656 | 1.047 | 0.001 | 0.002 | 1.71 s | 0.057 | 1050 MB |
| native CoreML — CPU+ANE† | 4.026 | 1.089 | 0.001 | 0.002 | 5.13 s | 0.171 | 1181 MB |
| PyTorch — MPS | — | — | — | — | 1.43 s | 0.048 | 1832 MB |
| PyTorch — CPU (reference) | — | — | — | — | 18.04 s | 0.602 | 7474 MB |

(stage columns are seconds; RTF = processing_time / 30 s, lower is faster. Embedding
predicts are batched — see the "batching" perf note. †ANE row predates batching; ANE
isn't a recommended path — see below.)

**Takeaways (honest):**
- vs **PyTorch-CPU**: native CoreML+GPU is **~16× faster** (RTF 0.037 vs 0.602) and uses
  **~25× less memory** (298 MB vs 7.3 GB).
- vs **PyTorch-MPS**: native CoreML+GPU is now **faster** (0.037 vs 0.048) **and ~6× lighter**
  (298 MB vs 1832 MB) — and Python-free (a self-contained Swift binary + CoreML models,
  no torch/pyannote runtime). MPS also silently CPU-fallbacks some ops
  (`aten::_weight_norm_interface`).
- Pitch: **runs natively on macOS, fastest measured here, tiny footprint, no Python.**

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
- **Why ANE is unused (compute-plan proof).** `benchmarks/compute_plan.py` reads the
  CoreML `MLComputePlan` and tallies the preferred device per op (643 ops):
  under **CPU+GPU → 643/643 on the GPU**; under **CPU+ANE → 643/643 on the CPU, 0 on the
  Neural Engine.** CoreML's planner refuses to place any op of this model on the ANE —
  the segmentation is transformer-heavy (linear ×110, matmul ×28, layer_norm ×47,
  reshape ×106, transpose ×77) and several op types (linear, conv, gelu, reduce_mean)
  aren't ANE-supported in this form. The ANE is a convolution accelerator; the embedding
  **ResNet** (a conv net) *does* benefit from it, but the WavLM+Conformer segmentation
  does not. Getting it onto the ANE would need re-architecting the model in "ANE
  principles" form (4-D tensors, conv2d projections instead of linear, split attention
  heads — Apple's *Deploying Transformers on the Apple Neural Engine*), i.e. a model
  rewrite + weight re-map, not a conversion flag. Given CPU+GPU already hits RTF 0.037,
  that optimization is low-ROI and left as possible future work.
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
- **Energy (measured on M2 Pro, system-wide while the pipeline loops, machine idle).**
  Under the default `.cpuAndGPU`: **ANE 0 mW** throughout, GPU ~18 W, CPU ~1.6 W,
  ~19-20 W combined. The pipeline is GPU-bound and the ANE is idle, matching the
  compute-plan (643/643 on the GPU). Forcing `.cpuAndNeuralEngine` keeps **ANE at
  0 mW** (CoreML still refuses to place this model on the ANE) while CPU jumps to
  ~6-11 W and combined power rises to ~22-32 W, so it is both slower and less
  power-efficient. Reproduce with `sudo benchmarks/energy.sh <wav> [cpu-gpu|cpu-ane]`
  (needs `sudo` for `powermetrics`; figures are indicative, not per-process).
