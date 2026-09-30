# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow semantic versioning.

## [Unreleased]

## [0.1.1] - 2026-09-30

### Added
- Corpus accuracy. The port was scored over the full 16-meeting AMI-SDM test set
  (base 15.79%, large 13.76% DER), matching DiariZen's published ~15.8%, and over
  31 VoxConverse files with 12-21 speakers (base 7.24%, large 6.35% at collar
  0.25), where it finds 8-23 speakers and never caps the count. New
  `validation/eval_corpus.py`.
- Compiled-model caching: `CoreMLBackend` caches the `.mlmodelc` next to the
  `.mlpackage`, so a host that spawns the CLI per file no longer recompiles it
  every run.
- `scripts/export-models.sh [base|large]` for one-command model export.
- Measured energy numbers in `RESULTS.md` (the ANE draws 0 W under the default
  policy).
- README infographics generated from source: pipeline diagram, benchmark panels,
  stat tiles, the example timeline, and the AMI and VoxConverse accuracy charts.
- swift-format lint in CI, `CONTRIBUTING.md`, issue and pull-request templates,
  Dependabot, and a prebuilt `diarizen-cli` binary attached to the release.

### Fixed
- `benchmarks/energy.sh` crashed under macOS bash 3.2 on the default `cpu-gpu`
  path (empty-array expansion under `set -u`).

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

[Unreleased]: https://github.com/NikiO-INO/diarizen-apple/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/NikiO-INO/diarizen-apple/releases/tag/v0.1.1
[0.1.0]: https://github.com/NikiO-INO/diarizen-apple/releases/tag/v0.1.0
