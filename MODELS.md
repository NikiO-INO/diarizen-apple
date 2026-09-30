# Model weights & licensing

This repository ships **conversion code and a runtime, not model weights.**

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
  and carry the same non-commercial restriction. Generate them locally; do not
  redistribute them under a permissive license.
- The intended distribution model when this repo goes public: users clone the
  code, accept DiariZen's license on Hugging Face, download the checkpoint, and
  run the conversion themselves. Attribution to DiariZen (BUT Speech@FIT) and the
  underlying pyannote/WavLM work is required.

If a commercially-usable diarizer is ever needed, a different (permissively
licensed) model would have to be substituted — the pipeline and wrappers here are
model-agnostic by design.

## Exporting the models to CoreML

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
