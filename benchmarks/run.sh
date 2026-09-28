#!/usr/bin/env bash
# Reproduce the benchmark numbers in RESULTS.md.
#
#   benchmarks/run.sh [audio.wav]
#
# Requires: a release build, the exported models in build/coreml/ (Segmentation +
# Embedding .mlpackage, plda_transform.json), and the DiariZen venv for the PyTorch
# reference. Peak memory via `/usr/bin/time -l` (macOS: bytes).
set -euo pipefail
cd "$(dirname "$0")/.."

WAV="${1:-build/DiariZen/example/EN2002a_30s.wav}"
RUNS="${RUNS:-10}"

echo "=== building release ==="
swift build -c release >/dev/null

echo
echo "=== native CoreML — CPU ==="
/usr/bin/time -l .build/release/diarizen-cli "$WAV" --models build/coreml \
    --compute-units cpu --benchmark "$RUNS" 2>&1 | grep -E "backend|segmentation|embedding|clustering|reconstruction|total|RTF|maximum resident"

echo
echo "=== native CoreML — CPU+GPU (default; ANE deadlocks, excluded) ==="
/usr/bin/time -l .build/release/diarizen-cli "$WAV" --models build/coreml \
    --benchmark "$RUNS" 2>&1 | grep -E "backend|segmentation|embedding|clustering|reconstruction|total|RTF|maximum resident"

echo
echo "=== PyTorch CPU (reference) ==="
HF_HOME="$PWD/build/hf-cache" .venv/bin/python benchmarks/bench_python.py --wav "$WAV" --runs 5
