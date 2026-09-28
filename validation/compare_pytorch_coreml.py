"""Assert PyTorch ≈ CoreML for the native (production) segmentation model.

    python validation/compare_pytorch_coreml.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --coreml build/coreml/ \
        --fixture validation/fixtures/parity_16k.wav

Parity is measured DECISION-LEVEL, in probability space plus per-frame argmax
agreement — not on the raw log-softmax logits. The segmentation head emits
log-probs reaching ~-14; the CoreML runtime diverges most in that deep-negative
tail, which exp() maps to ~0 and which changes no diarization decision (verified:
FP32 diverges as much as FP16, so it is runtime numerics, not rounding or a
conversion bug — the graph is structurally faithful, proven by the 1e-5 ONNX
match). See _parity.compare_logprob_segmentation / COREML_PROB_ATOL.

COMPUTE UNITS: we load CPU_ONLY on purpose. `compute_units=ALL` (the default)
DEADLOCKS on this WavLM+Conformer model during the first ANE predict on Apple
Silicon — the process hangs indefinitely at ~0% CPU. Until that ANE path is
profiled (Phase 4), validation and the Swift runtime must not blind-select ALL.
Runs on macOS only (needs the CoreML runtime).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import compare_logprob_segmentation, load_fixture_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--fixture", default="validation/fixtures/parity_16k.wav")
    args = ap.parse_args()

    import coremltools as ct
    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    mods = extract_modules(load_pipeline(args.model))
    audio = load_fixture_audio(args.fixture, mods.sample_rate)
    win = audio[: mods.seg_window_samples]
    x = torch.from_numpy(win).reshape(1, 1, -1)

    with torch.no_grad():
        ref = mods.segmentation.eval()(x).cpu().numpy()

    # CPU_ONLY: ALL/ANE deadlocks the first predict on this model (see module docstring).
    mlmodel = ct.models.MLModel(
        os.path.join(args.coreml, "Segmentation.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    out = mlmodel.predict({"waveform": x.numpy()})
    # ML Program output name defaults to the traced output; grab the sole tensor.
    got = next(iter(out.values()))

    result = compare_logprob_segmentation("segmentation (PyTorch vs CoreML)", ref, got)
    print(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
