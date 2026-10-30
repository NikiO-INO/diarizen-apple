# Model weights & licensing

This repository ships **conversion code and a runtime, not model weights.**
There are two ways to get the CoreML models the runtime needs:

1. **Download them ready-made** from the gated Hugging Face repositories below
   (accept the CC BY-NC 4.0 terms on the model page first).
2. **Convert them yourself** from the upstream checkpoints
   ([Exporting the models to CoreML](#exporting-the-models-to-coreml)).

## Ready-made CoreML models (gated)

| Repository | Converted from | Notes |
|---|---|---|
| [`nikioth/diarizen-wavlm-base-s80-md-coreml`](https://huggingface.co/nikioth/diarizen-wavlm-base-s80-md-coreml) | `BUT-FIT/diarizen-wavlm-base-s80-md` | the tensor-faithful default |
| [`nikioth/diarizen-wavlm-large-s80-md-v2-coreml`](https://huggingface.co/nikioth/diarizen-wavlm-large-s80-md-v2-coreml) | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | best accuracy; see the tensor-divergence note in the README |

Each repository holds exactly the three files `--models` expects
(`Segmentation.mlpackage`, `Embedding.mlpackage`, `plda_transform.json`), the
export scripts, and a model card with the measured accuracy and speed. They are
the same files the benchmarks in `benchmarks/RESULTS.md` were measured with.
Access is gated: you agree to the non-commercial terms below before the
download starts.

```bash
hf auth login                      # once; the repositories are gated
hf download nikioth/diarizen-wavlm-base-s80-md-coreml --local-dir build/coreml
hf download nikioth/diarizen-wavlm-large-s80-md-v2-coreml --local-dir build/coreml-large
```

## Upstream checkpoints (downloaded by the user)

DiariZen checkpoints live on the Hugging Face Hub under BUT Speech@FIT:

| Checkpoint | Notes |
|---|---|
| `BUT-FIT/diarizen-wavlm-base-s80-md` | smaller/faster — the bring-up target |
| `BUT-FIT/diarizen-wavlm-large-s80-md` | large |
| `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | latest large (best accuracy) |

## License of the weights: **CC BY-NC 4.0**

The DiariZen **model weights** are licensed **CC BY-NC 4.0 — non-commercial,
research/academic use only** (a restriction inherited from the training-data
licensing). The upstream card states: *"By downloading these weights, you agree
to use them for research and academic purposes only."*

The **DiariZen code** is MIT; **this repository's code** is MIT (see LICENSE).

### What this means

- **Do not commit weights** (`.pt`, `.ckpt`, `.safetensors`, `.onnx`, `.mlpackage`,
  `.mlmodelc`) to this repo — they are git-ignored.
- Converted artifacts (ONNX / CoreML) are **derivatives of the CC-BY-NC weights**
  and carry the same non-commercial restriction. Never redistribute them under a
  permissive license; the ready-made repositories above are published under
  CC BY-NC 4.0 behind a terms-acceptance gate for this reason.
- Two ways to get the models: download the gated CoreML conversions above, or
  clone the code, accept DiariZen's license on Hugging Face, download the
  checkpoint, and run the conversion yourself. Either way, attribution to DiariZen
  (BUT Speech@FIT) and the underlying pyannote/WavLM/WeSpeaker work is required.
- Within a conversion, `Embedding.mlpackage` (WeSpeaker ResNet34-LM, packaged by
  pyannote) and DiariZen's `plda/` files are CC BY 4.0 on their own; the
  segmentation model is what makes the set non-commercial.

If a commercially-usable diarizer is ever needed, a different (permissively
licensed) model would have to be substituted — the pipeline and wrappers here are
model-agnostic by design.

## Exporting the models to CoreML

The alternative to the ready-made downloads.

The quickest path is `scripts/export-models.sh base` (or `large`), which runs the
three steps below into the right directory. To do it by hand, export the three
artifacts a checkpoint needs into one directory (base and large each get their own).

```bash
# base-s80-md -> build/coreml/
python conversion/export_coreml.py           --model BUT-FIT/diarizen-wavlm-base-s80-md --out build/coreml
python conversion/export_embedding_coreml.py --model BUT-FIT/diarizen-wavlm-base-s80-md --out build/coreml
python conversion/export_plda.py             --model BUT-FIT/diarizen-wavlm-base-s80-md --out build/coreml/plda_transform.json

# large-s80-md-v2 -> build/coreml-large/
python conversion/export_coreml.py           --model BUT-FIT/diarizen-wavlm-large-s80-md-v2 --out build/coreml-large
python conversion/export_embedding_coreml.py --model BUT-FIT/diarizen-wavlm-large-s80-md-v2 --out build/coreml-large
python conversion/export_plda.py             --model BUT-FIT/diarizen-wavlm-large-s80-md-v2 --out build/coreml-large/plda_transform.json
```

Each directory ends up with:

```
build/coreml/
├── Segmentation.mlpackage
├── Embedding.mlpackage
└── plda_transform.json
```

Point the CLI at a directory with `--models build/coreml`. `conversion/inspect_model.py`
prints a checkpoint's geometry (powerset classes, frame count) before you export.
