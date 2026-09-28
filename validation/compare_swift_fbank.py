"""Assert the SWIFT Kaldi fbank ≈ torchaudio kaldi (WeSpeaker `compute_fbank`).

Feeds the byte-identical [-1,1] window to both; each side scales by 32768, runs
its kaldi fbank (80 mel, 25/10 ms Hamming, preemph 0.97), and applies per-mel CMVN.
Compares the (frames, 80) features. Tolerance-based (vDSP FFT vs torch.fft differ
in the last digits) — this is the Level-1 check of the embedding frontend.

    swift build
    python validation/compare_swift_fbank.py \
        --cli .build/debug/diarizen-cli \
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
ATOL = 5e-2   # log-mel units (values span roughly ±20 after CMVN)
MEAN_TOL = 5e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
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

    # Kaldi reference via WeSpeaker's own frontend.
    wespeaker = load_pipeline(args.model)._embedding.model_.eval()
    with torch.no_grad():
        ref = wespeaker.compute_fbank(torch.from_numpy(window).reshape(1, 1, -1))[0].numpy()  # (1598, 80)

    # Swift fbank via the CLI dump.
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "window.f32")
        json_path = os.path.join(tmp, "fbank.json")
        window.tofile(raw_path)
        subprocess.run(
            [args.cli, "--raw-input", raw_path, "--dump-fbank", json_path],
            check=True,
        )
        with open(json_path) as f:
            got = np.asarray(json.load(f), dtype=np.float32)  # (1598, 80)

    if got.shape != ref.shape:
        print(f"[FAIL] shape mismatch: swift {got.shape} vs kaldi {ref.shape}")
        return 1

    diff = np.abs(ref - got)
    mx, mean = float(diff.max()), float(diff.mean())
    passed = mx <= ATOL and mean <= MEAN_TOL
    print(f"[{'PASS' if passed else 'FAIL'}] fbank (Swift vs kaldi): "
          f"max_abs={mx:.3e} mean_abs={mean:.3e} (atol {ATOL}, mean_tol {MEAN_TOL})")
    # Where is the worst frame, for debugging?
    if not passed:
        fi = int(diff.max(axis=1).argmax())
        print(f"    worst frame {fi}: swift[:4]={got[fi][:4]}  kaldi[:4]={ref[fi][:4]}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
