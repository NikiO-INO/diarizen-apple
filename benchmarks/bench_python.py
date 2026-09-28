"""Reference benchmark: the upstream DiariZen (PyTorch, CPU) pipeline.

Measures inference wall-clock (model loading excluded) and peak RSS on the same
fixture the native Swift benchmark uses, so RTF is directly comparable. PyTorch on
CPU is the honest reference here — the Swift port targets CoreML, and DiariZen's
compute_fbank falls back to CPU for FFT anyway.

    python benchmarks/bench_python.py --runs 5
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    import torch
    import torchaudio

    from _diarizen_loader import load_pipeline

    torch.set_num_threads(os.cpu_count() or 1)
    pipe = load_pipeline(args.model)  # not timed

    info = torchaudio.info(args.wav)
    audio_dur = info.num_frames / info.sample_rate

    pipe(args.wav, sess_name="warmup")  # warm-up (not timed)

    runs = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        pipe(args.wav, sess_name="bench")
        runs.append(time.perf_counter() - t0)
    runs.sort()
    median = runs[len(runs) // 2]

    # macOS ru_maxrss is in bytes.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print(f"backend: PyTorch CPU; runs={args.runs}; audio={audio_dur:.1f}s")
    print(f"  total (median) {median:.3f}s")
    print(f"  RTF            {median / audio_dur:.4f}")
    print(f"  peak RSS       {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
