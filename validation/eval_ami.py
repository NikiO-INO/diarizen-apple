"""Accuracy check on a full AMI meeting against the human reference.

Runs the upstream DiariZen (PyTorch) pipeline on a long real file, writes its RTTM,
and reports DER against the manual reference — the "what DiariZen achieves" baseline
that the Swift port aims to reproduce. Also reports the train-embedding count (drives
Swift AHC cost).

    python validation/eval_ami.py \
        --wav build/ami/EN2002a.Array1-01.wav \
        --ref build/ami/EN2002a.ref.rttm \
        --sess EN2002a --out build/ami/EN2002a.pyannote.rttm
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--wav", default="build/ami/EN2002a.Array1-01.wav")
    ap.add_argument("--ref", default="build/ami/EN2002a.ref.rttm")
    ap.add_argument("--sess", default="EN2002a")
    ap.add_argument("--out", default="build/ami/EN2002a.pyannote.rttm")
    args = ap.parse_args()

    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    from _diarizen_loader import load_pipeline

    # Count train embeddings that reach clustering (drives Swift AHC cost).
    n_train = {}
    from pyannote.audio.pipelines.clustering import BaseClustering
    orig = BaseClustering.filter_embeddings
    def spy(self, embeddings, segmentations=None, min_frames_ratio=0.1):
        out = orig(self, embeddings, segmentations=segmentations, min_frames_ratio=min_frames_ratio)
        n_train["n"] = out[0].shape[0]
        return out
    BaseClustering.filter_embeddings = spy

    pipe = load_pipeline(args.model)
    t0 = time.perf_counter()
    result = pipe(args.wav, sess_name=args.sess)
    dt = time.perf_counter() - t0

    with open(args.out, "w") as f:
        f.write(result.to_rttm())

    ref = next(iter(load_rttm(args.ref).values()))
    der = DiarizationErrorRate()(ref, result)

    n_ref_spk = len(ref.labels())
    n_hyp_spk = len(result.labels())
    print(f"AMI {args.sess}: pyannote(PyTorch) DER vs reference = {der*100:.2f}%")
    print(f"  wall={dt:.1f}s  ref speakers={n_ref_spk}  hyp speakers={n_hyp_spk}")
    print(f"  train embeddings into clustering (Swift AHC size) = {n_train.get('n')}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
