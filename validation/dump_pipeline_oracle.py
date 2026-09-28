"""Dump pyannote pipeline intermediates as the Swift-aggregation oracle.

Runs the real DiariZenPipeline up to (but not including) clustering and saves the
Phase-3 inputs — sliding-window geometry, binarized per-chunk segmentations, and
per-(chunk, speaker) embeddings — so the Swift aggregation/embedding wiring can be
validated against them.

    python validation/dump_pipeline_oracle.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --wav build/DiariZen/example/EN2002a_30s.wav \
        --out build/oracle/EN2002a.npz

Writes build/oracle/*.npz (git-ignored, model-derived).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))


def _make_coreml_forward(mlpackage):
    """Drop-in segmentation forward backed by the exported CoreML model (CPU), so
    the oracle uses the SAME segmentation as the Swift port — isolating the Swift
    aggregation/embedding logic from the CoreML-vs-PyTorch segmentation delta."""
    import coremltools as ct
    import numpy as np
    import torch

    ml = ct.models.MLModel(mlpackage, compute_units=ct.ComputeUnit.CPU_ONLY)
    W = 256_000

    def forward(waveforms, *args, **kwargs):
        x = waveforms.detach().cpu().numpy().astype(np.float32)
        b, _c, t = x.shape
        if t != W:
            pad = np.zeros((b, x.shape[1], W), dtype=np.float32)
            pad[:, :, : min(t, W)] = x[:, :, : min(t, W)]
            x = pad
        outs = [ml.predict({"waveform": x[i : i + 1, :1, :]})["segmentation"] for i in range(b)]
        return torch.from_numpy(np.concatenate(outs, axis=0)).to(waveforms.device)

    return forward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--out", default="build/oracle/EN2002a.npz")
    ap.add_argument("--segmentation", choices=["pytorch", "coreml"], default="pytorch",
                    help="coreml: swap the CoreML seg into pyannote so the oracle matches Swift")
    ap.add_argument("--coreml", default="build/coreml")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchaudio
    from scipy.ndimage import median_filter
    from pyannote.audio.core.inference import SlidingWindowFeature

    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)

    if args.segmentation == "coreml":
        import os as _os
        pkg = _os.path.join(args.coreml, "Segmentation.mlpackage")
        pipe._segmentation.model.forward = _make_coreml_forward(pkg)
        print(f"using CoreML segmentation from {pkg}")

    waveform, sr = torchaudio.load(args.wav)
    waveform = torch.unsqueeze(waveform[0], 0)  # SDM single channel, (1, N)
    file = {"waveform": waveform, "sample_rate": sr}

    seg = pipe.get_segmentations(file, soft=False)  # SWF (num_chunks, 799, num_speakers)
    sw = seg.sliding_window
    data = seg.data
    if pipe.apply_median_filtering:
        data = median_filter(data, size=(1, 11, 1), mode="reflect")
    binarized = SlidingWindowFeature(data, sw)

    embeddings = pipe.get_embeddings(
        file, binarized, exclude_overlap=pipe.embedding_exclude_overlap
    )  # (num_chunks, num_speakers, dim)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        sw_start=np.float64(sw.start),
        sw_duration=np.float64(sw.duration),
        sw_step=np.float64(sw.step),
        num_samples=np.int64(waveform.shape[1]),
        sample_rate=np.int64(sr),
        binarized=data.astype(np.float32),      # (num_chunks, 799, num_speakers)
        embeddings=embeddings.astype(np.float32),  # (num_chunks, num_speakers, dim)
    )
    print(f"wrote {args.out}")
    print(f"  sliding window: start={sw.start:.4f} duration={sw.duration:.4f} step={sw.step:.4f}")
    print(f"  binarized: {data.shape}  embeddings: {embeddings.shape}")
    print(f"  num_samples={waveform.shape[1]} sr={sr}")
    # sanity: NaN embeddings appear for inactive speakers (empty mask)
    nan_rows = int(np.isnan(embeddings).any(axis=-1).sum())
    print(f"  embedding rows all-NaN (inactive speakers): {nan_rows}/{embeddings.shape[0]*embeddings.shape[1]}")


if __name__ == "__main__":
    main()
