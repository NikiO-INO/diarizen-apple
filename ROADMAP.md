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

## Phase 1 — ONNX reference backend  ✅ DONE
- [x] `export_onnx.py`: segmentation (WavLM+Conformer) → `segmentation.onnx`
      (opset 17, static shapes). Exports cleanly.
- [x] Runs under ONNX Runtime CPU.
- [x] `compare_pytorch_onnx.py`: **PyTorch ≈ ONNX PASS** — max_abs=3.05e-05,
      mean_abs=4.0e-06 on the `parity_16k` fixture (FP32 rounding).
- [x] `reproduce_rttm.py`: wire ONNX segmentation into the FULL upstream pipeline
      (WeSpeaker embeddings + VBx clustering) by swapping only
      `pipeline._segmentation.model.forward`. On `EN2002a_30s` (3 speakers, overlap):
      **DER(golden, onnx)=0.000%, byte-identical RTTM.** Golden RTTM saved to
      `validation/fixtures/EN2002a.golden.rttm` for the Swift port to diff against.
- **Gate:** ✅ segmentation parity; ✅ full-pipeline RTTM reproduction.

## Phase 2 — Native CoreML production backend  🟢 segmentation converted + validated
- [x] `conversion/export_coreml.py`: convert **directly PyTorch → CoreML**
      (`ct.convert(..., convert_to="mlprogram")`), FP16, static single-chunk shape.
      Needed two export-only patches (`conversion/patches/wavlm_export_patch.py`):
      neutralize WavLM's `@torch.jit.export` variable-length `layer_norm` (un-scriptable)
      and drop `GradMultiply` (a `torch.autograd.Function` → un-convertible `pythonop`,
      identity at inference). Both are no-ops for our fixed mono chunk.
- [x] `validation/compare_pytorch_coreml.py`: **PyTorch ≈ CoreML PASS** —
      prob_max=2.1e-2, prob_mean=7.8e-4, **argmax_agree=99.87%** on `parity_16k.wav`.
      Parity is measured DECISION-LEVEL (probability space + per-frame argmax), not on
      raw log-softmax logits: the runtime diverges ~0.02-0.2 only in the deep-negative
      log-prob tail (exp→~0, decision-irrelevant). FP32 diverges as much as FP16, so it
      is runtime numerics, not FP16 rounding and not a conversion bug (graph is faithful —
      the ONNX check matches to 1e-5). See `_parity.compare_logprob_segmentation`.
- [x] ⚠️ **ANE deadlock found & documented:** `compute_units=.all` hangs the first
      Neural Engine predict indefinitely (~0% CPU) on this WavLM+Conformer model; CPU/GPU
      predict returns in ms. Validation loads `.cpuOnly`; the Swift `ComputePolicy` default
      is now `.cpuAndGPU`, with ANE opt-in/experimental until Phase 4 profiling.
- [ ] `Sources/DiariZen/CoreMLBackend.swift` + model wrappers load the `.mlmodelc`
      and run inference; `Pipeline.swift` wires segmentation → aggregation →
      embeddings → clustering. (Backend policy done; wrappers still skeletons.)
- **Gate:** ✅ segmentation converts + reproduces PyTorch decisions on CPU/GPU;
  ⬜ full Swift pipeline reproduces the upstream RTTM end-to-end; ⬜ ANE path fixed.

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
