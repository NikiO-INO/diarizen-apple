"""Dump the full-pipeline RTTM (CoreML segmentation) as the Swift end-to-end oracle,
plus the segmentation frame resolution the Swift reconstruction needs.

Runs the real DiariZenPipeline with the CoreML segmentation swapped in (so the
oracle matches the Swift port's segmentation), and writes:
  - build/oracle/EN2002a.coreml.rttm   (the reference RTTM the Swift CLI should reproduce)
  - build/oracle/frames.json           (frame SlidingWindow: start/duration/step)

    python validation/dump_rttm_oracle.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from dump_pipeline_oracle import _make_coreml_forward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--out-rttm", default="build/oracle/EN2002a.coreml.rttm")
    ap.add_argument("--out-frames", default="build/oracle/frames.json")
    args = ap.parse_args()

    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)
    pipe._segmentation.model.forward = _make_coreml_forward(
        os.path.join(args.coreml, "Segmentation.mlpackage")
    )

    result = pipe(args.wav, sess_name="EN2002a")
    rttm = result.to_rttm()

    rf = pipe._segmentation.model._receptive_field
    frames = {"start": float(rf.start), "duration": float(rf.duration), "step": float(rf.step)}

    os.makedirs(os.path.dirname(args.out_rttm), exist_ok=True)
    with open(args.out_rttm, "w") as f:
        f.write(rttm)
    with open(args.out_frames, "w") as f:
        json.dump(frames, f)

    print(f"wrote {args.out_rttm}")
    print(f"wrote {args.out_frames}: {frames}")
    print("--- RTTM ---")
    print(rttm.rstrip())


if __name__ == "__main__":
    main()
