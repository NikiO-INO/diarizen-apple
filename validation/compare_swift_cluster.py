"""Validate the Swift PLDA transform + VBx against the DiariZen oracle.

Feeds Swift the SAME train embeddings + AHC init the Python VBx used
(build/oracle/*_cluster.npz) and compares the PLDA-space features, the VBx
responsibilities `gamma`, and the priors `pi`. Isolates the clustering math from
AHC and from the embedding delta.

    swift build
    python validation/dump_clustering_oracle.py         # writes the oracle
    python validation/compare_swift_cluster.py \
        --cli .build/debug/diarizen-cli \
        --coreml build/coreml \
        --oracle build/oracle/EN2002a_cluster.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

FEA_ATOL = 1e-3
GAMMA_ATOL = 1e-3
PI_ATOL = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--oracle", default="build/oracle/EN2002a_cluster.npz")
    args = ap.parse_args()

    import numpy as np

    if not os.path.exists(args.cli):
        print(f"CLI not found at {args.cli} — run `swift build` first.", file=sys.stderr)
        return 2

    o = np.load(args.oracle)
    train_emb, ahc = o["train_emb"], o["ahc"]
    Fa, Fb, maxIters = float(o["Fa"]), float(o["Fb"]), int(o["maxIters"])
    ref_fea, ref_gamma, ref_pi = o["fea"], o["gamma"], o["pi"]

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.json")
        out_path = os.path.join(tmp, "out.json")
        with open(in_path, "w") as f:
            json.dump({"train_emb": train_emb.astype(float).tolist(),
                       "ahc": ahc.astype(int).tolist(),
                       "Fa": Fa, "Fb": Fb, "maxIters": maxIters}, f)
        subprocess.run([args.cli, "--models", args.coreml,
                        "--debug-cluster", in_path, "--output", out_path], check=True)
        got = json.load(open(out_path))
    got_fea = np.asarray(got["fea"], dtype=np.float64)
    got_gamma = np.asarray(got["gamma"], dtype=np.float64)
    got_pi = np.asarray(got["pi"], dtype=np.float64)

    ok = True
    def check(name, ref, gotv, atol):
        nonlocal ok
        if gotv.shape != ref.shape:
            print(f"[FAIL] {name}: shape swift {gotv.shape} vs oracle {ref.shape}"); ok = False; return
        mx = float(np.abs(ref - gotv).max())
        p = mx <= atol; ok = ok and p
        print(f"[{'PASS' if p else 'FAIL'}] {name}: max_abs={mx:.3e} (atol {atol})")

    check("fea (PLDA transform)", ref_fea, got_fea, FEA_ATOL)
    check("gamma (VBx responsibilities)", ref_gamma, got_gamma, GAMMA_ATOL)
    check("pi (VBx priors)", ref_pi, got_pi, PI_ATOL)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
