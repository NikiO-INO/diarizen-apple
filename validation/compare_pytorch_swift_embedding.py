"""Level-3 (end-to-end) embedding parity:

    Swift fbank + CoreML ResNet   ≈   kaldi fbank + PyTorch ResNet

Feeds the byte-identical [-1,1] window + the same per-frame weights to both full
stacks and compares the 256-d embeddings by cosine similarity (tolerance-based —
this composes the Level-1 fbank delta with the Level-2 FP16 ResNet delta).

    swift build
    python validation/compare_pytorch_swift_embedding.py \
        --cli .build/debug/diarizen-cli \
        --coreml build/coreml \
        --fixture validation/fixtures/parity_16k.wav
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import load_fixture_audio  # noqa: E402

CROP_SAMPLES = 256_000
SEG_FRAMES = 799
COS_MIN = 0.999   # composes fbank (~2e-4) + FP16 ResNet deltas


def cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--fixture", default="validation/fixtures/parity_16k.wav")
    args = ap.parse_args()

    import numpy as np
    import torch

    from _diarizen_loader import load_pipeline

    if not os.path.exists(args.cli):
        print(f"CLI not found at {args.cli} — run `swift build` first.", file=sys.stderr)
        return 2

    audio = load_fixture_audio(args.fixture, 16_000)
    window = np.ascontiguousarray(audio[:CROP_SAMPLES], dtype=np.float32)
    if window.shape[0] < CROP_SAMPLES:
        window = np.pad(window, (0, CROP_SAMPLES - window.shape[0]))

    wespeaker = load_pipeline(args.model)._embedding.model_.eval()
    with torch.no_grad():
        fbank = wespeaker.compute_fbank(torch.from_numpy(window).reshape(1, 1, -1))

    masks = {
        "ones": np.ones(SEG_FRAMES, dtype=np.float32),
        "block": np.concatenate([np.ones(400, np.float32), np.zeros(SEG_FRAMES - 400, np.float32)]),
    }

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "window.f32")
        window.tofile(raw_path)

        for name, w in masks.items():
            with torch.no_grad():
                ref = wespeaker.resnet(fbank, torch.from_numpy(w).reshape(1, -1))[1].cpu().numpy()

            cmd = [args.cli, "--models", args.coreml, "--compute-units", "cpu",
                   "--raw-input", raw_path, "--dump-embedding", os.path.join(tmp, f"{name}.json")]
            if name != "ones":
                wpath = os.path.join(tmp, f"{name}.w32")
                w.tofile(wpath)
                cmd += ["--weights-raw", wpath]
            subprocess.run(cmd, check=True)
            with open(os.path.join(tmp, f"{name}.json")) as f:
                got = np.asarray(json.load(f), dtype=np.float32)

            cos = cosine(ref, got)
            mx = float(np.abs(ref.ravel() - got.ravel()).max())
            passed = cos >= COS_MIN
            ok = ok and passed
            print(f"[{'PASS' if passed else 'FAIL'}] e2e embedding[{name}]: "
                  f"cosine={cos:.6f} max_abs={mx:.3e} (cos≥{COS_MIN})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
