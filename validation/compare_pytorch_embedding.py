"""Assert PyTorch ≈ CoreML for the WeSpeaker ResNet34 embedding neural part.

Feeds the byte-identical (CMVN'd fbank, weights) to both backends and compares the
256-d embeddings by cosine similarity + max abs (tolerance-based, not bit-identical,
because CoreML runs FP16). The PyTorch side uses the ORIGINAL `StatsPool.forward`
(torch.vmap), so this ALSO validates the export-time StatsPool patch is a no-op.

The fbank is produced here by the real `compute_fbank` (kaldi) — the Swift fbank
frontend is validated separately (compare_swift_fbank / end-to-end embedding parity).

    python validation/compare_pytorch_embedding.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --coreml build/coreml \
        --fixture validation/fixtures/parity_16k.wav
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import load_fixture_audio  # noqa: E402

CROP_SAMPLES = 256_000
SEG_FRAMES = 799
COS_MIN = 0.9999          # FP16 embedding vs FP32 reference
EMB_ATOL = 5e-2           # per-component; embeddings are O(1)-O(10) magnitude


def cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--fixture", default="validation/fixtures/parity_16k.wav")
    args = ap.parse_args()

    import coremltools as ct
    import numpy as np
    import torch

    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)
    wespeaker = pipe._embedding.model_.eval()

    audio = load_fixture_audio(args.fixture, 16_000)
    crop = np.ascontiguousarray(audio[:CROP_SAMPLES], dtype=np.float32)
    if crop.shape[0] < CROP_SAMPLES:
        crop = np.pad(crop, (0, CROP_SAMPLES - crop.shape[0]))
    wav = torch.from_numpy(crop).reshape(1, 1, -1)

    with torch.no_grad():
        fbank = wespeaker.compute_fbank(wav)  # (1, 1598, 80), CMVN'd

    ml = ct.models.MLModel(
        os.path.join(args.coreml, "Embedding.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # Test a couple of weight patterns: fully-active + a block mask (exercises the
    # interpolation + masked pooling path).
    masks = {
        "ones": torch.ones(1, SEG_FRAMES),
        "block": torch.cat([torch.ones(1, 400), torch.zeros(1, SEG_FRAMES - 400)], dim=1),
    }

    ok = True
    for name, weights in masks.items():
        with torch.no_grad():
            ref = wespeaker.resnet(fbank, weights)[1].cpu().numpy()  # original vmap forward
        got = ml.predict({"fbank": fbank.numpy(), "weights": weights.numpy()})["embedding"]
        cos = cosine(ref, got)
        mx = float(np.abs(ref.ravel() - np.asarray(got).ravel()).max())
        passed = cos >= COS_MIN and mx <= EMB_ATOL
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] embedding[{name}]: cosine={cos:.6f} "
              f"max_abs={mx:.3e} (cos≥{COS_MIN}, atol {EMB_ATOL})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
