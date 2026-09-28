"""Export DiariZen's neural sub-modules to ONNX (the REFERENCE backend).

Role: fast bring-up, prove WavLM + Conformer export cleanly, and produce a
portable reference to validate the native CoreML backend against. This is NOT
the final Apple runtime.

    python conversion/export_onnx.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --out build/onnx/

Phase 1 deliverable. Requires the DiariZen env (see requirements.txt).
"""

from __future__ import annotations

import argparse
import os

OPSET = 17


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--out", default="build/onnx")
    ap.add_argument("--opset", type=int, default=OPSET)
    args = ap.parse_args()

    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    os.makedirs(args.out, exist_ok=True)
    pipe = load_pipeline(args.model)
    mods = extract_modules(pipe)

    seg = mods.segmentation.eval()
    # DiariZen's segmentation forward asserts a 3-D input (batch, channel, sample)
    # and selects one channel internally; output is (batch, frame, classes).
    dummy = torch.zeros(1, mods.export_channels, mods.seg_window_samples, dtype=torch.float32)

    seg_path = os.path.join(args.out, "segmentation.onnx")
    torch.onnx.export(
        seg,
        (dummy,),
        seg_path,
        input_names=["waveform"],
        output_names=["segmentation"],
        opset_version=args.opset,
        # Keep dynamic axes MINIMAL — only what's genuinely variable. Fixed shapes
        # help both ORT and the later CoreML conversion.
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f"wrote {seg_path}")

    if mods.embedding is not None:
        # TODO(phase1): export the embedding model too, with its real input shape.
        print("embedding model present — TODO: export in phase 1")
    else:
        print("embedding fused into segmentation — nothing separate to export")

    print(
        "\nNext: `python validation/compare_pytorch_onnx.py --model",
        args.model,
        "--onnx",
        args.out,
        "` to check numerical parity.",
    )


if __name__ == "__main__":
    main()
