"""Assert PyTorch ≈ CoreML for the native (production) segmentation model.

    python validation/compare_pytorch_coreml.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --coreml build/coreml/ \
        --fixture validation/fixtures/EN2002a_30s.wav

CoreML runs FP16 by default, so tolerances are looser than the ONNX check
(_parity.COREML_*). If parity fails, first confirm it isn't just FP16 by
re-exporting fp32 and comparing. Runs on macOS only (needs the CoreML runtime).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversion"))

from _parity import COREML_ATOL, COREML_RTOL, compare, load_fixture_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--fixture", default="validation/fixtures/EN2002a_30s.wav")
    args = ap.parse_args()

    import coremltools as ct
    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    mods = extract_modules(load_pipeline(args.model))
    audio = load_fixture_audio(args.fixture, mods.sample_rate)
    win = audio[: mods.seg_window_samples]
    x = torch.from_numpy(win).unsqueeze(0)

    with torch.no_grad():
        ref = mods.segmentation.eval()(x).cpu().numpy()

    mlmodel = ct.models.MLModel(os.path.join(args.coreml, "Segmentation.mlpackage"))
    out = mlmodel.predict({"waveform": x.numpy()})
    # ML Program output name defaults to the traced output; grab the sole tensor.
    got = next(iter(out.values()))

    result = compare("segmentation (PyTorch vs CoreML)", ref, got, COREML_ATOL, COREML_RTOL)
    print(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
