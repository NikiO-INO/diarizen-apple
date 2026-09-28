# diarizen-apple

On-device [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) speaker
diarization for **Apple Silicon**, with a native CoreML production backend and an
ONNX reference/validation backend.

DiariZen (BUT Speech@FIT) is, as of 2026, the strongest **open-source** speaker
diarization pipeline — a WavLM-Large front-end with powerset segmentation and VBx
clustering (~13.3% DER on the multilingual benchmark in
[arXiv:2509.26177](https://arxiv.org/abs/2509.26177), ahead of vanilla
pyannote 3.1 and unlike Sortformer/Nemotron it is **not capped at a small number
of speakers**). It ships only as a PyTorch/Python pipeline. This project makes it
run efficiently on a Mac, offline, with no Python at inference time.

> **Status:** scaffolding / bring-up. Nothing here produces final numbers yet.
> See [ROADMAP.md](ROADMAP.md) for the phased plan.

---

## Design principles

1. **Two backends, two roles — not one "chosen" backend.**

   ```
                     PyTorch checkpoint
                          /        \
                       ONNX       CoreML
                        |            |
                   reference     production
                    backend      Apple backend
   ```

   - **ONNX = reference / compatibility.** `torch.onnx.export` of the original
     model, run under ONNX Runtime (CPU, and CoreML Execution Provider). Its job:
     fast bring-up, prove WavLM + Conformer export cleanly, give us numerical
     reference outputs, run regression/parity tests, and stay portable beyond
     Apple. **ORT + CoreML EP is NOT the final Apple runtime** — it can partition
     the graph, silently fall back to CPU, and handle dynamic shapes poorly.

   - **CoreML = production Apple Silicon backend.** Converted **directly from
     PyTorch via `coremltools`** (not PyTorch→ONNX→CoreML — no extra lowering
     layer unless technically forced). ML Program, FP16 where safe,
     fixed/enumerated input shapes where possible, profiled on the Neural Engine,
     Swift API, configurable compute units (`.all`, `.cpuAndNeuralEngine`, …).

2. **MLX is a *possible future* experimental backend — not the current path.** It
   would mean re-implementing WavLM + Conformer, mapping weights, reproducing the
   structured pruning, possibly custom ops, and re-validating everything —
   large surface area and numerical-drift risk. CoreML is the more natural ANE
   production target today.

3. **The pipeline is modular — not one giant graph.** Neural parts stay separate
   models; glue stays in Swift/CPU:

   ```
   audio
     → [CoreML] WavLM + Conformer segmentation
     → [Swift]  chunk aggregation
     → [CoreML] speaker embeddings
     → [Swift]  clustering / VBx / post-processing
     → diarization result (RTTM)
   ```

4. **Reproducible conversion + parity tests are first-class.** Golden tensors and
   test audio with explicit tolerances assert `PyTorch ≈ ONNX` and
   `PyTorch ≈ CoreML` (see [`validation/`](validation/)).

5. **Honest benchmarks.** We measure RTF, latency, peak memory, CPU, and (where
   possible) energy across ONNX Runtime CPU / ONNX Runtime CoreML EP / native
   CoreML / PyTorch-MPS and across Apple Silicon generations. The README will not
   claim "runs fully on ANE" unless profiling confirms it — the honest phrasing
   is *"CoreML backend optimized for Apple Neural Engine."*

---

## Licensing (read before distributing)

- **This project's code:** MIT (see [LICENSE](LICENSE)).
- **DiariZen model weights:** the upstream checkpoints are **CC BY-NC 4.0
  (non-commercial / research use only)**. See [MODELS.md](MODELS.md).

Because of that, **weights are never committed to this repository.** The flow is:

```
official checkpoint (HuggingFace, CC BY-NC)
    → downloaded / user-provided
    → conversion script (this repo)
    → generated ONNX / CoreML model (user's machine)
```

We ship the conversion code and the runtime; the user fetches the weights
themselves under their license.

---

## Repository layout

```
diarizen-apple/
├── Sources/
│   ├── DiariZen/            # Swift library: the production CoreML pipeline
│   │   ├── Audio/           # decode + 16 kHz mono resampling
│   │   ├── Segmentation/    # WavLM+Conformer segmentation wrapper
│   │   ├── Embeddings/      # speaker-embedding model wrapper
│   │   ├── Clustering/      # VBx / AHC + post-processing (CPU)
│   │   ├── CoreMLBackend.swift
│   │   └── Pipeline.swift
│   └── diarizen-cli/        # thin CLI: audio → RTTM (used by hosts + benchmarks)
├── conversion/             # PyTorch → ONNX and PyTorch → CoreML exporters
│   ├── export_onnx.py
│   ├── export_coreml.py
│   └── patches/            # upstream tracing/export patches, if any
├── validation/             # parity harness (PyTorch ≈ ONNX ≈ CoreML)
│   ├── compare_pytorch_onnx.py
│   ├── compare_pytorch_coreml.py
│   └── fixtures/           # test audio + golden tensors (git-ignored if large)
├── benchmarks/
│   └── RESULTS.md
└── Tests/                  # Swift tests
```

## Integration with meetlify

The `diarizen-cli` executable takes an audio file and writes an RTTM hypothesis,
mirroring `fluidaudiocli nemotron3-diarize`. meetlify plugs it in as another
selectable diarization engine (sidecar pattern) once the CoreML backend passes
parity + benchmarks.

## Quick start (developer)

```bash
# 1. Python env for conversion + validation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Export ONNX reference (downloads the CC-BY-NC checkpoint)
python conversion/export_onnx.py --model BUT-FIT/diarizen-wavlm-base-s80-md --out build/onnx/

# 3. Check PyTorch ≈ ONNX parity
python validation/compare_pytorch_onnx.py --model BUT-FIT/diarizen-wavlm-base-s80-md --onnx build/onnx/

# 4. Export native CoreML directly from PyTorch
python conversion/export_coreml.py --model BUT-FIT/diarizen-wavlm-base-s80-md --out build/coreml/

# 5. Check PyTorch ≈ CoreML parity, then build the Swift runtime
python validation/compare_pytorch_coreml.py --model BUT-FIT/diarizen-wavlm-base-s80-md --coreml build/coreml/
swift build -c release
```
