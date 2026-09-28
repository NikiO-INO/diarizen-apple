"""Dump VBx clustering intermediates as the Swift-clustering oracle.

Runs DiariZen's VBxClustering stages on the (CoreML-segmentation) feature oracle
and saves each intermediate so the Swift port can be validated stage by stage:
train embeddings + their (chunk, speaker) origin, AHC init labels, PLDA-space
features, VBx responsibilities/priors, and the final hard clusters.

    python validation/dump_clustering_oracle.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --features build/oracle/EN2002a.npz \
        --out build/oracle/EN2002a_cluster.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--features", default="build/oracle/EN2002a.npz")
    ap.add_argument("--out", default="build/oracle/EN2002a_cluster.npz")
    args = ap.parse_args()

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import cdist
    from pyannote.audio.core.inference import SlidingWindow, SlidingWindowFeature

    from diarizen.clustering.VBx import cluster_vbx, vbx_setup
    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)
    cl = pipe.clustering

    feats = np.load(args.features)
    binarized = feats["binarized"]        # (C, 799, S)
    embeddings = feats["embeddings"]      # (C, S, 256)
    seg = SlidingWindowFeature(binarized, SlidingWindow(start=0.0, duration=16.0, step=1.6))

    # 1. filter_embeddings (min_frames_ratio=0.1)
    train_emb, cidx, sidx = cl.filter_embeddings(embeddings, segmentations=seg, min_frames_ratio=0.1)

    # 2. AHC (centroid linkage, distance threshold 0.6)
    normed = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    dendro = linkage(normed, method="centroid", metric="euclidean")
    ahc = fcluster(dendro, cl.ahc_threshold, criterion=cl.ahc_criterion) - 1
    _, ahc = np.unique(ahc, return_inverse=True)

    # 3. PLDA transform + VBx
    x_tf, plda_tf, plda_psi = vbx_setup(cl.plda_dir)
    fea = plda_tf(x_tf(train_emb), lda_dim=cl.lda_dim)
    Phi = plda_psi[: cl.lda_dim]
    gamma, pi = cluster_vbx(ahc, fea, Phi, Fa=cl.Fa, Fb=cl.Fb, maxIters=cl.maxIters)

    # 4. full clustering result (hard_clusters) via the real __call__
    hard_clusters, soft_clusters, centroids = cl(embeddings, seg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        train_emb=train_emb.astype(np.float32),
        cidx=cidx.astype(np.int32), sidx=sidx.astype(np.int32),
        ahc=ahc.astype(np.int32),
        fea=fea.astype(np.float64),
        Phi=Phi.astype(np.float64),
        gamma=gamma.astype(np.float64), pi=pi.astype(np.float64),
        hard_clusters=hard_clusters.astype(np.int32),
        Fa=np.float64(cl.Fa), Fb=np.float64(cl.Fb), maxIters=np.int64(cl.maxIters),
    )
    print(f"wrote {args.out}")
    print(f"  train_emb {train_emb.shape}  ahc {ahc.shape} (clusters={ahc.max()+1})")
    print(f"  fea {fea.shape}  gamma {gamma.shape}  pi {pi.shape}")
    print(f"  hard_clusters {hard_clusters.shape} -> speakers {sorted(set(hard_clusters.ravel().tolist()))}")


if __name__ == "__main__":
    main()
