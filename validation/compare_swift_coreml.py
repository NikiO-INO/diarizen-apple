"""Assert the SWIFT CoreML segmentation == the PYTHON CoreML segmentation.

Both backends load the same `.mlpackage` and run on the same CPU compute units,
fed the byte-identical input window (raw Float32, bypassing each side's audio
loader). So the outputs should match to ~float epsilon — this validates the Swift
inference path (MLMultiArray packing, feature names, output decoding) against the
already-PyTorch-validated Python path.

    swift build            # produce .build/debug/diarizen-cli
    python validation/compare_swift_coreml.py \
        --cli .build/debug/diarizen-cli \
        --coreml build/coreml \
        --fixture validation/fixtures/parity_16k.wav

macOS only. Run after export_coreml.py has written build/coreml/Segmentation.mlpackage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import compare_logprob_segmentation, load_fixture_audio  # noqa: E402

WINDOW_SAMPLES = 256_000  # must match CoreMLSegmentation.windowSamples
# Same model + same CPU runtime + identical input → expect near bit-identical.
RAW_ATOL = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--fixture", default="validation/fixtures/parity_16k.wav")
    args = ap.parse_args()

    import numpy as np
    import coremltools as ct

    if not os.path.exists(args.cli):
        print(f"CLI not found at {args.cli} — run `swift build` first.", file=sys.stderr)
        return 2

    audio = load_fixture_audio(args.fixture, 16_000)
    window = np.ascontiguousarray(audio[:WINDOW_SAMPLES], dtype=np.float32)
    if window.shape[0] < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - window.shape[0]))

    # --- Python CoreML reference (CPU) ---
    ml = ct.models.MLModel(
        os.path.join(args.coreml, "Segmentation.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    ref = ml.predict({"waveform": window.reshape(1, 1, -1)})["segmentation"][0]  # (799, 11)

    # --- Swift CoreML via the CLI dump mode, fed the identical raw window ---
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "window.f32")
        json_path = os.path.join(tmp, "seg.json")
        window.tofile(raw_path)  # little-endian float32 on Apple Silicon
        subprocess.run(
            [args.cli, "--models", args.coreml, "--compute-units", "cpu",
             "--raw-input", raw_path, "--dump-segmentation", json_path],
            check=True,
        )
        with open(json_path) as f:
            got = np.asarray(json.load(f), dtype=np.float32)  # (799, 11)

    if got.shape != ref.shape:
        print(f"[FAIL] shape mismatch: swift {got.shape} vs python {ref.shape}")
        return 1

    raw_max = float(np.abs(ref - got).max())
    result = compare_logprob_segmentation("segmentation (Swift vs Python CoreML)", ref, got)
    raw_ok = raw_max <= RAW_ATOL
    print(result)
    print(f"    raw logit max_abs={raw_max:.3e}  (tol {RAW_ATOL:.0e}) -> {'PASS' if raw_ok else 'FAIL'}")
    return 0 if (result.passed and raw_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
