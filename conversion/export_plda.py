"""Export the VBx PLDA transform as fixed constants for the Swift clustering.

`diarizen.clustering.VBx.vbx_setup` derives, from the checkpoint's PLDA files, the
transforms that map a 256-d speaker embedding into the 128-d PLDA space where VBx
runs. Those transforms are FIXED (they include a generalized eigendecomposition,
`scipy.linalg.eigh(B, W)`), so we compute them once here and ship the resulting
matrices — Swift then needs only linear algebra, no eigensolver.

Emits build/coreml/plda_transform.json (git-ignored, model-derived / CC-BY-NC):
    mean1 (256), lda (256x128), mean2 (128)          # xvec_tf (centering/whitening/LDA)
    plda_mu (128), plda_tr (128x128), plda_psi (128) # plda_tf + Phi (post-eigh, reordered)

    python conversion/export_plda.py --model BUT-FIT/diarizen-wavlm-base-s80-md
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--out", default="build/coreml/plda_transform.json")
    args = ap.parse_args()

    import numpy as np
    from scipy.linalg import eigh

    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)
    tf_dir = pipe.clustering.plda_dir

    x = np.load(os.path.join(tf_dir, "xvec_transform.npz"))
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]

    p = np.load(os.path.join(tf_dir, "plda.npz"))
    plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]

    # Reproduce vbx_setup exactly: W/B, generalized eigendecomposition, reorder.
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    def arr(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return {"shape": list(a.shape), "data": a.ravel().tolist()}

    payload = {
        "mean1": arr(mean1), "mean2": arr(mean2), "lda": arr(lda),
        "plda_mu": arr(plda_mu), "plda_tr": arr(plda_tr), "plda_psi": arr(plda_psi),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}")
    for k, v in payload.items():
        print(f"  {k}: {v['shape']}")


if __name__ == "__main__":
    main()
