# Roadmap

Phased plan. Each phase has a **gate** (a concrete, checkable exit criterion)
before moving on. Start on the smaller `diarizen-wavlm-base-s80-md` checkpoint;
upgrade to `diarizen-wavlm-large-s80-md-v2` (best accuracy) through the same
wrappers once the pipeline is proven.

## Phase 0 — Environment & upstream understanding
- [ ] Pin a working Python env (torch 2.1.1, coremltools, onnx, onnxruntime, the
      DiariZen + custom pyannote fork). Record exact versions in `requirements.txt`.
- [ ] Run upstream `DiariZenPipeline.from_pretrained(...)` on `example/*.wav`;
      capture the reference RTTM and intermediate tensors.
- [ ] Map the pipeline: locate the segmentation `nn.Module`, the embedding model,
      the chunking/aggregation logic, and the clustering (VBx/AHC) code paths.
      Document exact input/output shapes and dtypes in `docs/PIPELINE.md`.
- **Gate:** we can run DiariZen end-to-end in Python and have named the exact
  modules + tensor shapes to export.

## Phase 1 — ONNX reference backend
- [ ] `conversion/export_onnx.py`: export the segmentation model (WavLM+Conformer)
      and the embedding model to ONNX (opset ≥ 17), with dynamic axes only where
      genuinely needed.
- [ ] Run under ONNX Runtime CPU.
- [ ] `validation/compare_pytorch_onnx.py`: assert `PyTorch ≈ ONNX` on golden
      fixtures within tolerances (start `atol=1e-3, rtol=1e-3`, tighten later).
- **Gate:** ONNX segmentation + embedding outputs match PyTorch within tolerance
  on all fixtures; full ONNX+Swift(or Python)-clustering pipeline reproduces the
  upstream RTTM.

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
