"""Corpus DER over a directory of {meeting}.ref.rttm + {meeting}.<tag>.rttm pairs.

    python validation/eval_corpus.py --dir build/ami --tag swift

Scores each meeting's hypothesis RTTM against its reference with pyannote's
DiarizationErrorRate() defaults (collar=0, skip_overlap=False, no UEM) — the same
metric as validation/eval_ami.py — and accumulates into one metric object, so the
reported corpus DER is duration-weighted (micro-averaged) over the whole set, not
a plain mean of per-meeting percentages.

Hypotheses come from the Swift CLI, e.g.:

    for r in build/ami/*.ref.rttm; do m=$(basename "$r" .ref.rttm)
      .build/release/diarizen-cli "build/ami/$m.Array1-01.wav" \
        --models build/coreml --output "build/ami/$m.swift.rttm"; done
"""

from __future__ import annotations

import argparse
import glob
import os


def _load(path: str):
    from pyannote.database.util import load_rttm

    return next(iter(load_rttm(path).values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="build/ami")
    ap.add_argument("--tag", default="swift", help="hypothesis suffix: {meeting}.<tag>.rttm")
    args = ap.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate()
    rows = []
    for ref_path in sorted(glob.glob(os.path.join(args.dir, "*.ref.rttm"))):
        meeting = os.path.basename(ref_path)[: -len(".ref.rttm")]
        hyp_path = os.path.join(args.dir, f"{meeting}.{args.tag}.rttm")
        if not os.path.exists(hyp_path):
            print(f"skip {meeting}: no {args.tag} hypothesis")
            continue
        ref, hyp = _load(ref_path), _load(hyp_path)
        der = metric(ref, hyp)
        rows.append((meeting, der * 100.0, len(ref.labels()), len(hyp.labels())))

    for meeting, der, n_ref, n_hyp in rows:
        print(f"{meeting:10s} DER={der:6.2f}%  ref_spk={n_ref}  hyp_spk={n_hyp}")
    print(f"\nCorpus DER ({args.tag}) = {abs(metric) * 100.0:.2f}%  over {len(rows)} meetings")


if __name__ == "__main__":
    main()
