"""Print the geometry a DiariZen checkpoint implies, so the Swift constants
(window samples, powerset shape, frame resolution, seg frames, embedding, PLDA)
can be verified/updated before exporting a new model.

    python conversion/inspect_model.py --model BUT-FIT/diarizen-wavlm-large-s80-md-v2
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    args = ap.parse_args()

    import numpy as np
    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    pipe = load_pipeline(args.model)
    mods = extract_modules(pipe)
    seg = mods.segmentation

    with torch.no_grad():
        out = seg(torch.zeros(1, mods.export_channels, mods.seg_window_samples))
    spec = seg.specifications
    rf = seg._receptive_field

    print(f"=== {args.model} ===")
    print(f"segmentation:")
    print(f"  window_samples = {mods.seg_window_samples}  (seg_duration {mods.seg_window_samples/mods.sample_rate:.1f}s)")
    print(f"  output shape   = {tuple(out.shape)}  (frames={out.shape[1]}, powerset_classes={out.shape[2]})")
    print(f"  powerset: num_classes={getattr(spec, 'num_powerset_classes', '?')} "
          f"max_set_size={getattr(spec, 'powerset_max_classes', '?')} "
          f"classes(speakers)={len(spec.classes) if hasattr(spec,'classes') else '?'}")
    print(f"  receptive_field: start={rf.start} duration={rf.duration} step={rf.step}")

    emb = pipe._embedding
    print(f"embedding:")
    print(f"  class={type(emb.model_).__name__} dim={emb.dimension} sr={emb.sample_rate}")

    cl = pipe.clustering
    print(f"clustering: {type(cl).__name__} lda_dim={cl.lda_dim} Fa={cl.Fa} Fb={cl.Fb} "
          f"ahc_thr={cl.ahc_threshold} maxIters={cl.maxIters}")
    d = cl.plda_dir
    for fn in ["xvec_transform.npz", "plda.npz"]:
        z = np.load(os.path.join(d, fn))
        print(f"  {fn}: {{ {', '.join(f'{k}:{tuple(z[k].shape)}' for k in z.files)} }}")


if __name__ == "__main__":
    main()
