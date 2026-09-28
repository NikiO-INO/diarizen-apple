#!/usr/bin/env bash
# Rough energy / power profile of the native pipeline.
#
#   sudo benchmarks/energy.sh [audio.wav] [compute-units]
#
# `powermetrics` needs root and samples SYSTEM-WIDE power, so keep the machine
# otherwise idle. We run the pipeline in a tight loop and sample CPU/GPU/ANE power
# concurrently; the reported figures are indicative, not per-process attribution.
# Use it mainly to confirm which engines light up (e.g. ANE power ~0 under the
# default cpu-gpu policy).
set -euo pipefail
cd "$(dirname "$0")/.."

WAV="${1:-build/DiariZen/example/EN2002a_30s.wav}"
UNITS="${2:-cpu-gpu}"

if [[ "$UNITS" == "cpu-gpu" ]]; then UNIT_ARG=(); else UNIT_ARG=(--compute-units "$UNITS"); fi

echo "loop: diarizen-cli $WAV (units=$UNITS) ×200 in background"
.build/release/diarizen-cli "$WAV" --models build/coreml "${UNIT_ARG[@]}" --benchmark 200 >/dev/null 2>&1 &
LOOP_PID=$!

# ~10 s of samples at 500 ms.
powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 20 2>/dev/null \
    | grep -iE "CPU Power|GPU Power|ANE Power|Combined Power|package Power" || true

kill "$LOOP_PID" 2>/dev/null || true
echo "done (samples above are system-wide while the pipeline looped)"
