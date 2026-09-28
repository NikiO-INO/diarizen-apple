# Roadmap

Phased plan. Each phase has a **gate** (a concrete, checkable exit criterion)
before moving on. Start on the smaller `diarizen-wavlm-base-s80-md` checkpoint;
upgrade to `diarizen-wavlm-large-s80-md-v2` (best accuracy) through the same
wrappers once the pipeline is proven.

## Phase 0 — Environment & upstream understanding  ✅ DONE
- [x] Python env: uv + **Python 3.11 + torch 2.1.1 + torchaudio 2.1.1 +
      accelerate 0.29** (NOT transformers 5.x — forces torch ≥ 2.5) + the DiariZen
      pyannote fork (editable). `diarizen` installed `--no-deps` + a few runtime
      deps (psutil, librosa, soundfile, toml, huggingface_hub).
- [x] Loaded `diarizen-wavlm-base-s80-md`; ran a forward.
- [x] Mapped the pipeline; exact shapes/dtypes in `docs/PIPELINE.md`:
      segmentation `(1,1,256000) → (1,799,11)`, VBx clustering, WeSpeaker embed.
- **Gate met:** DiariZen runs in Python; export target + shapes are named.

## Phase 1 — ONNX reference backend  🟢 segmentation done
- [x] `export_onnx.py`: segmentation (WavLM+Conformer) → `segmentation.onnx`
      (opset 17, static shapes). Exports cleanly.
- [x] Runs under ONNX Runtime CPU.
- [x] `compare_pytorch_onnx.py`: **PyTorch ≈ ONNX PASS** — max_abs=3.05e-05,
      mean_abs=4.0e-06 on the `EN2002a_30s` fixture (FP32 rounding).
- [ ] Wire ONNX segmentation into the full pipeline (embedding + VBx) and
      reproduce the upstream RTTM end-to-end.
- **Gate:** ✅ segmentation parity; ⬜ full-pipeline RTTM reproduction.

## Phase 2 — Native CoreML production backend
- [ ] `conversion/export_coreml.py`: convert **directly PyTorch → CoreML**
      (`coremltools`, `ct.convert(..., convert_to="mlprogram")`), FP16 where safe,
      fixed or enumerated input shapes. No ONNX in this path.
- [ ] `validation/compare_pytorch_coreml.py`: assert `PyTorch ≈ CoreML` within
      tolerance (CoreML FP16 will need looser tol than ONNX FP32 — document it).
- [ ] `Sources/DiariZen/CoreMLBackend.swift` + model wrappers load the `.mlmodelc`
      and run inference; `Pipeline.swift` wires segmentation → aggregation →
      embeddings → clustering.
- **Gate:** native CoreML pipeline reproduces the upstream RTTM within a small DER
  delta on the parity fixtures; runs on ANE-capable compute units.

## Phase 3 — Clustering & post-processing in Swift
- [ ] Port VBx / AHC + powerset decoding + overlap handling to Swift/CPU (or reuse
      an existing implementation), validated against the Python clustering on the
      same segmentation outputs.
- **Gate:** Swift clustering matches Python clustering on identical inputs.

## Phase 4 — CLI + benchmarks
- [ ] `Sources/diarizen-cli`: `diarizen-cli <audio.wav> [--compute-units …] --output <rttm>`.
- [ ] `benchmarks/`: RTF / latency / peak memory / CPU / energy across PyTorch-MPS,
      ORT CPU, ORT CoreML EP, native CoreML, on ≥1 Apple Silicon generation.
      Fill `benchmarks/RESULTS.md`. Do **not** claim "runs on ANE" without profiling.
- **Gate:** reproducible benchmark numbers + honest README claims.

## Phase 5 — meetlify integration
- [ ] Add a `diarizen` diarization engine to meetlify (sidecar over `diarizen-cli`,
      RTTM parsing already exists via `turns_from_rttm`).
- **Gate:** selectable in Settings, A/B-able against Nemotron/FluidAudio.

## Phase 6 — upgrade + open-source
- [ ] Swap the wrappers to `diarizen-wavlm-large-s80-md-v2`; re-run parity + benchmarks.
- [ ] Polish docs, CONTRIBUTING, CI; flip the repo public.

## Later (not now)
- MLX experimental backend (re-implementation, weight mapping, revalidation).
