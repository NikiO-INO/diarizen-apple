"""Validate the Swift Phase-2 aggregation against the pyannote oracle.

Compares the Swift-produced pre-clustering features (binarized per-chunk
segmentations + per-(chunk, speaker) embeddings) against build/oracle/*.npz.

Generate the oracle with `--segmentation coreml` so pyannote uses the SAME CoreML
segmentation as the Swift port — that isolates the Swift aggregation/embedding
LOGIC (sliding window, powerset decode, median filter, exclude-overlap masking)
from the CoreML-vs-PyTorch segmentation delta. Remaining differences are then only
Swift fbank vs kaldi + FP16 ResNet on identical masks (the Level-3 delta).

    swift build
    python validation/dump_pipeline_oracle.py --segmentation coreml  # writes the oracle
    python validation/compare_swift_features.py \
        --cli .build/debug/diarizen-cli \
        --coreml build/coreml \
        --wav build/DiariZen/example/EN2002a_30s.wav \
        --oracle build/oracle/EN2002a.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

BIN_AGREE_MIN = 0.999  # same segmentation → only powerset/median logic differs
EMB_COS_MIN = 0.999    # identical masks → only Swift fbank + FP16 ResNet (Level-3) delta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--oracle", default="build/oracle/EN2002a.npz")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchaudio

    if not os.path.exists(args.cli):
        print(f"CLI not found at {args.cli} — run `swift build` first.", file=sys.stderr)
        return 2
    if not os.path.exists(args.oracle):
        print(f"oracle not found at {args.oracle} — run dump_pipeline_oracle.py first.", file=sys.stderr)
        return 2

    oracle = np.load(args.oracle)
    ref_bin = oracle["binarized"]        # (C, 799, S)
    ref_emb = oracle["embeddings"]       # (C, S, D)

    # Feed Swift the exact samples the oracle saw (channel 0, no resample).
    waveform, _ = torchaudio.load(args.wav)
    samples = np.ascontiguousarray(waveform[0].numpy(), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "audio.f32")
        feat_path = os.path.join(tmp, "features.json")
        samples.tofile(raw_path)
        subprocess.run(
            [args.cli, "--models", args.coreml, "--compute-units", "cpu",
             "--raw-input", raw_path, "--dump-features", feat_path],
            check=True,
        )
        with open(feat_path) as f:
            feats = json.load(f)
    got_bin = np.asarray(feats["binarized"], dtype=np.float32)   # (C, 799, S)
    got_emb = np.asarray(feats["embeddings"], dtype=np.float32)  # (C, S, D)

    ok = True

    # --- binarized segmentations ---
    if got_bin.shape != ref_bin.shape:
        print(f"[FAIL] binarized shape: swift {got_bin.shape} vs oracle {ref_bin.shape}")
        return 1
    agree = float(((got_bin > 0.5) == (ref_bin > 0.5)).mean())
    b_ok = agree >= BIN_AGREE_MIN
    ok = ok and b_ok
    print(f"[{'PASS' if b_ok else 'FAIL'}] binarized: cell_agreement={agree*100:.3f}% "
          f"(≥{BIN_AGREE_MIN*100:.0f}%)  shape={got_bin.shape}")

    # --- embeddings (cosine per chunk-speaker; only where the oracle speaker is active) ---
    if got_emb.shape != ref_emb.shape:
        print(f"[FAIL] embeddings shape: swift {got_emb.shape} vs oracle {ref_emb.shape}")
        return 1
    C, S, _ = ref_emb.shape
    cosines = []
    active_cos = []
    for c in range(C):
        for s in range(S):
            a, b = ref_emb[c, s], got_emb[c, s]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(a @ b / (na * nb + 1e-12))
            cosines.append(cos)
            if ref_bin[c, :, s].sum() > 0:  # speaker actually active in this chunk
                active_cos.append(cos)
    mean_cos = float(np.mean(active_cos)) if active_cos else 1.0
    min_cos = float(np.min(active_cos)) if active_cos else 1.0
    e_ok = mean_cos >= EMB_COS_MIN
    ok = ok and e_ok
    print(f"[{'PASS' if e_ok else 'FAIL'}] embeddings: active slots={len(active_cos)}/{C*S} "
          f"mean_cos={mean_cos:.5f} min_cos={min_cos:.5f} (≥{EMB_COS_MIN})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
