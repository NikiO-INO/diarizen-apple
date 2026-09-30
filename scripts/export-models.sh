#!/usr/bin/env bash
# Export a DiariZen checkpoint to the three CoreML artifacts the runtime needs.
#
#   scripts/export-models.sh [base|large] [out_dir]
#
# Downloads the CC BY-NC weights from Hugging Face (accept the license on the
# model page first) and writes Segmentation.mlpackage, Embedding.mlpackage, and
# plda_transform.json into out_dir. Needs the Python env from requirements.txt
# (set PYTHON=... to pick a specific interpreter, e.g. PYTHON=.venv/bin/python).
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:-base}"
case "$VARIANT" in
  base)  MODEL="BUT-FIT/diarizen-wavlm-base-s80-md";     DEF_OUT="build/coreml" ;;
  large) MODEL="BUT-FIT/diarizen-wavlm-large-s80-md-v2"; DEF_OUT="build/coreml-large" ;;
  *) echo "usage: $0 [base|large] [out_dir]" >&2; exit 2 ;;
esac
OUT="${2:-$DEF_OUT}"
PY="${PYTHON:-python3}"

echo "== exporting $MODEL -> $OUT =="
"$PY" conversion/export_coreml.py           --model "$MODEL" --out "$OUT"
"$PY" conversion/export_embedding_coreml.py --model "$MODEL" --out "$OUT"
"$PY" conversion/export_plda.py             --model "$MODEL" --out "$OUT/plda_transform.json"

echo "== done. $OUT now has: =="
ls -1 "$OUT"
