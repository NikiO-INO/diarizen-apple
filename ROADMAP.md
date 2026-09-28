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

## Phase 2 — Native CoreML production backend  ✅ neural stages + aggregation done (clustering → Phase 3)
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
- [x] `CoreMLBackend.swift` loads an `.mlpackage` (compiles on the fly) or a
      `.mlmodelc`; default `ComputePolicy` = `.cpuAndGPU` (ANE opt-in).
- [x] `CoreMLSegmentation.segment` runs the model: packs `[Float]` → `(1,1,256000)`
      `MLMultiArray` (feature `waveform`), decodes output `segmentation` `(1,799,11)`
      → `[frames][classes]`. **Swift CoreML == Python CoreML, BIT-IDENTICAL**
      (`validation/compare_swift_coreml.py`: raw max_abs=0.0, argmax_agree=100% on the
      same raw input, CPU). Transitively PyTorch ≈ Swift-CoreML. `diarizen-cli` gains
      `--dump-segmentation`/`--raw-input` for this parity; unit tests cover the decode.
- [x] **Embedding stage** — kaldi fbank in Swift, ResNet34 in CoreML (decided split):
      - `export_embedding_coreml.py` converts `forward(fbank(1,1598,80), weights(1,799))
        → embedding(1,256)`; export patch `wespeaker_export_patch.py` swaps StatsPool's
        `torch.vmap` (un-traceable, emits `movedim`) for the single-speaker equivalent.
      - `KaldiFbank.swift` (Accelerate/vDSP): 80-mel/25/10 ms Hamming, preemph 0.97, DC
        removal, power spectrum, log, per-mel CMVN — DiariZen owns the frontend.
      - `CoreMLEmbedding.embed` = Swift fbank → CoreML ResNet.
      - **3-level parity (tolerance/cosine):** (1) Swift fbank ≈ kaldi max_abs=1.9e-4;
        (2) CoreML ResNet ≈ PyTorch cosine=0.99991; (3) end-to-end Swift-fbank+CoreML ≈
        kaldi+PyTorch cosine=0.99991. Scripts: `compare_swift_fbank.py`,
        `compare_pytorch_embedding.py`, `compare_pytorch_swift_embedding.py`.
- [x] **Aggregation** — `Pipeline.swift extractFeatures` reproduces pyannote's
      get_segmentations + get_embeddings up to clustering:
      - sliding window (16 s / 1.6 s step, unfold + zero-padded last chunk),
      - `Powerset.swift` decode (799×11 log-probs → argmax → 799×4 multi-label),
      - median filter (1,11,1) + half-sample-symmetric reflect,
      - per-(chunk,speaker) embeddings with exclude-overlap masking (clean-frame
        gate + min-frames fallback).
      Validated against a CoreML-segmentation oracle (same seg as Swift, isolating
      the aggregation logic): **binarized cell-agreement 100.000%**, **embeddings
      mean_cos 0.99991 / min 0.99980**. `dump_pipeline_oracle.py --segmentation coreml`
      + `compare_swift_features.py`; unit tests cover powerset/median/reflect.
- [x] Clustering + reconstruction → RTTM — see Phase 3, full end-to-end gate met.
- **Gate:** ✅ segmentation + embedding stages run from Swift and match Python;
  ✅ Swift aggregation reproduces pyannote's pre-clustering features exactly;
  ✅ full Swift RTTM (Phase 3, DER 0.495%); ⬜ ANE path fixed.

## Phase 3 — Clustering & post-processing in Swift  ✅ DONE
- [x] **VBx core** (`PLDA.swift`, `VBx.swift`): PLDA transform (constants from
      `export_plda.py`, eigh pre-solved) + the loopProb=0 GMM VBx. Bit-exact vs the
      oracle: fea 6.9e-14, gamma 5.7e-17, pi 1.1e-27.
- [x] **AHC** (`AHC.swift`): centroid linkage + distance-criterion flat cut → VBx init.
- [x] **Assignment** (`Clustering.swift`, `Hungarian.swift`): filter_embeddings
      (clean-frame gate), cluster centroids, cosine `cdist`, constrained argmax
      (max-weight assignment), renumber → hard_clusters.
- [x] **Reconstruction** (`Reconstruction.swift`): pyannote reconstruct →
      to_diarization (overlap-add `aggregate` + per-frame top-`count`) → `Binarize`.
- **Gate met:** full native Swift pipeline (CoreML seg → Swift fbank + CoreML embed →
  Swift VBx → reconstruct → RTTM) reproduces the pyannote (CoreML-seg) oracle at
  **DER 0.495%** (12↔12 turns) on EN2002a. `compare_swift_cluster.py`,
  `compare_swift_rttm.py`, `dump_clustering_oracle.py`, `dump_rttm_oracle.py`; unit
  tests cover Hungarian / AHC / cosine / Binarize.

## Phase 4 — CLI + benchmarks  ✅ DONE
- [x] `diarizen-cli <audio.wav> [--compute-units cpu-gpu|cpu|all|cpu-ane] [--output <rttm>]`
      emits RTTM; robust `AudioLoader` (fast path for 16 kHz mono). Plus `--benchmark`
      and the `--dump-*`/`--debug-cluster` parity modes.
- [x] `benchmarks/`: RTF / latency / peak memory across **native CoreML (CPU, CPU+GPU,
      CPU+ANE) vs PyTorch (CPU, MPS)** on M2 Pro. Native CPU+GPU: **RTF 0.055, 226 MB**.
      ~11× faster / ~33× lighter than PyTorch-CPU; a speed wash with PyTorch-MPS
      (0.048) but ~8× lighter and Python-free. `benchmarks/{run.sh,bench_python.py,RESULTS.md}`;
      per-stage timing via `Pipeline.timings`.
- [x] **ANE investigated (deadlock resolved):** `.all` (CPU+GPU+ANE) genuinely deadlocks
      (partitioning conflict — avoid). `.cpuAndNeuralEngine` works and is *accurate*
      (DER 0.505%) but WavLM seg is ~12× slower on ANE than GPU (transformer, not
      ANE-friendly). `.cpuAndGPU` stays the default. Docs corrected in `CoreMLBackend.swift`.
      Original 13-min "hang" = slow first-time ANE compilation, since cached.
- [x] Energy: `benchmarks/energy.sh` (sudo/`powermetrics`) provided for the user to run.
- [ ] Deferred (low value): ORT full-pipeline (only seg is in ONNX; not a production target).
- **Gate met:** reproducible benchmark numbers + honest claims; ANE understood, not overclaimed.

## Phase 5 — meetlify integration
- [ ] Add a `diarizen` diarization engine to meetlify (sidecar over `diarizen-cli`,
      RTTM parsing already exists via `turns_from_rttm`).
- **Gate:** selectable in Settings, A/B-able against Nemotron/FluidAudio.

## Phase 6 — upgrade + open-source
- [ ] Swap the wrappers to `diarizen-wavlm-large-s80-md-v2`; re-run parity + benchmarks.
- [ ] Polish docs, CONTRIBUTING, CI; flip the repo public.

## Later (not now)
- MLX experimental backend (re-implementation, weight mapping, revalidation).
