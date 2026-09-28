"""Assert PyTorch ≈ ONNX for the exported segmentation (and embedding) models.

    python validation/compare_pytorch_onnx.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --onnx build/onnx/ \
        --fixture validation/fixtures/EN2002a_30s.wav

Exit code is non-zero if any tensor is outside tolerance — wire this into CI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import ONNX_ATOL, ONNX_RTOL, compare, load_fixture_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--onnx", default="build/onnx")
    ap.add_argument("--fixture", default="validation/fixtures/EN2002a_30s.wav")
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    mods = extract_modules(load_pipeline(args.model))
    audio = load_fixture_audio(args.fixture, mods.sample_rate)
    # One segmentation window as (batch, channel, sample). Single-channel model.
    win = audio[: mods.seg_window_samples]
    x = torch.from_numpy(win).reshape(1, 1, -1)

    with torch.no_grad():
        ref = mods.segmentation.eval()(x).cpu().numpy()

    sess = ort.InferenceSession(
        os.path.join(args.onnx, "segmentation.onnx"),
        providers=["CPUExecutionProvider"],
    )
    got = sess.run(None, {"waveform": x.numpy().astype(np.float32)})[0]

    result = compare("segmentation (PyTorch vs ONNX-CPU)", ref, got, ONNX_ATOL, ONNX_RTOL)
    print(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
