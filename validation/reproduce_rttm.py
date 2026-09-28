"""End-to-end RTTM reproduction — the Phase 1 gate.

Proves the exported segmentation reproduces the upstream DiariZen diarization when
dropped into the FULL pipeline (WeSpeaker embeddings + VBx clustering), not just at
the tensor level. Runs the real `DiariZenPipeline` twice on one fixture:

  1. pure PyTorch segmentation  -> GOLDEN RTTM  (saved for the Swift port to diff against)
  2. ONNX-Runtime segmentation  -> candidate RTTM

and asserts DER(golden, candidate) is ~0. ONLY the neural segmentation forward is
swapped (`pipeline._segmentation.model.forward`); every downstream stage — powerset
conversion, median filtering, speaker counting, embeddings, clustering,
reconstruction, binarization — is the untouched upstream code. So any RTTM
difference is attributable to the export alone.

    python validation/reproduce_rttm.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --onnx build/onnx/segmentation.onnx \
        --wav build/DiariZen/example/EN2002a_30s.wav \
        --sess EN2002a \
        --golden-out validation/fixtures/EN2002a.golden.rttm

Runs on macOS/CPU; needs the DiariZen env (see requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

# Segmentation chunk geometry — must match the exported ONNX static shape.
SEG_WINDOW_SAMPLES = 256000  # 16 s * 16 kHz
# Golden vs ONNX should be near-identical; allow a hair for the 3e-5 segmentation
# delta flipping a boundary frame. Report the real number regardless.
DER_TOL = 0.01


def _make_onnx_forward(onnx_path):
    """Return a drop-in replacement for the segmentation model's forward that runs
    ONNX Runtime. Matches the model contract: (B, C, T) waveform -> (B, frame, class)
    log-probs. ONNX is static batch=1/mono, so we loop the batch and pad/select."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def onnx_forward(waveforms, *args, **kwargs):
        x = waveforms.detach().to("cpu").numpy().astype(np.float32)
        b, _c, t = x.shape
        if t != SEG_WINDOW_SAMPLES:  # pad/truncate a partial last chunk to the fixed window
            padded = np.zeros((b, x.shape[1], SEG_WINDOW_SAMPLES), dtype=np.float32)
            n = min(t, SEG_WINDOW_SAMPLES)
            padded[:, :, :n] = x[:, :, :n]
            x = padded
        outs = [sess.run(None, {"waveform": x[i : i + 1, :1, :]})[0] for i in range(b)]
        y = np.concatenate(outs, axis=0)
        return torch.from_numpy(y).to(waveforms.device)

    return onnx_forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--onnx", default="build/onnx/segmentation.onnx")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--sess", default="EN2002a")
    ap.add_argument("--golden-out", default="validation/fixtures/EN2002a.golden.rttm")
    args = ap.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    from _diarizen_loader import load_pipeline

    pipe = load_pipeline(args.model)

    # ---- 1. GOLDEN: pure PyTorch segmentation ----
    print("=== golden run (PyTorch segmentation) ===", flush=True)
    golden = pipe(args.wav, sess_name=args.sess)
    golden_rttm = golden.to_rttm()
    os.makedirs(os.path.dirname(args.golden_out), exist_ok=True)
    with open(args.golden_out, "w") as f:
        f.write(golden_rttm)
    print(f"wrote {args.golden_out}", flush=True)

    # ---- 2. CANDIDATE: swap in ONNX segmentation forward ----
    print("=== candidate run (ONNX segmentation) ===", flush=True)
    model = pipe._segmentation.model
    original_forward = model.forward
    model.forward = _make_onnx_forward(args.onnx)
    try:
        candidate = pipe(args.wav, sess_name=args.sess)
    finally:
        model.forward = original_forward
    candidate_rttm = candidate.to_rttm()

    # ---- 3. Compare ----
    der = DiarizationErrorRate()
    value = der(golden, candidate)  # golden = reference, candidate = hypothesis
    identical = golden_rttm.strip() == candidate_rttm.strip()

    print("\n--- GOLDEN RTTM (PyTorch) ---")
    print(golden_rttm.rstrip())
    print("\n--- CANDIDATE RTTM (ONNX) ---")
    print(candidate_rttm.rstrip())

    passed = value <= DER_TOL
    flag = "PASS" if passed else "FAIL"
    print(
        f"\n[{flag}] end-to-end RTTM: DER(golden, onnx)={value * 100:.3f}%  "
        f"byte_identical={identical}  (tol {DER_TOL * 100:.1f}%)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
