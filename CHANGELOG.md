# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow semantic versioning.

## [Unreleased]

## [0.1.0] - 2026-09-29

First working release of the Apple Silicon port.

### Added
- Native CoreML backends for segmentation (WavLM + Conformer) and speaker
  embeddings (WeSpeaker ResNet34), converted directly from the PyTorch
  checkpoints with `coremltools`.
- Swift Kaldi-compatible filterbank frontend on Accelerate/vDSP, so the mel
  features run without Python.
- Swift VBx clustering (PLDA plus a Bayesian HMM), agglomerative initialization,
  Hungarian assignment, and pyannote-style reconstruction to RTTM.
- `diarizen-cli`: audio file to RTTM, with `--models`, `--compute-units`,
  `--output`, and parity dump modes.
- Checkpoint-agnostic model loading. The powerset geometry is inferred from the
  segmentation output, so `base-s80-md` and `large-s80-md-v2` share one binary.
- Conversion scripts under `conversion/` and a three-level parity harness under
  `validation/`: Swift fbank against Kaldi, CoreML against PyTorch, and the full
  Swift pipeline against the upstream RTTM.
- Benchmarks under `benchmarks/` with reproducible real-time-factor, latency,
  and memory numbers, plus a CoreML compute-plan reader and an energy probe.

### Notes
- `base-s80-md` is the tensor-faithful default (30 s parity clip at 0.495% DER,
  full AMI EN2002a at 21.16%). `large-s80-md-v2` is more accurate (17.50%) with a
  documented, tolerated CoreML segmentation divergence.
- The default compute policy is CPU plus GPU. The Neural Engine is not used for
  this model; `benchmarks/RESULTS.md` has the compute-plan evidence.

[Unreleased]: https://github.com/NikiO-INO/diarizen-apple/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NikiO-INO/diarizen-apple/releases/tag/v0.1.0
