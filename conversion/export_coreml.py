"""Export DiariZen's neural sub-modules to native CoreML (the PRODUCTION Apple
backend), converting DIRECTLY from PyTorch — not via ONNX.

Rationale: no extra lowering layer unless technically forced. coremltools traces
the PyTorch module (via torch.jit) and emits an ML Program we run from Swift on
the Neural Engine.

    python conversion/export_coreml.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --out build/coreml/ \
        --precision fp16

Phase 2 deliverable. coremltools >= 8, and its torch-compat pin, are the fragile
parts — if coremltools conflicts with DiariZen's torch==2.1.1, run this in a
separate env from the ONNX/validation steps.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--out", default="build/coreml")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    import coremltools as ct
    import torch

    from _diarizen_loader import extract_modules, load_pipeline

    os.makedirs(args.out, exist_ok=True)
    pipe = load_pipeline(args.model)
    mods = extract_modules(pipe)

    seg = mods.segmentation.eval()

    # WavLM's @torch.jit.export'd variable-length path is un-scriptable; neutralize
    # it before tracing (see conversion/patches/wavlm_export_patch.py). No-op for
    # our fixed single-chunk input; parity tests confirm outputs are unchanged.
    from patches.wavlm_export_patch import apply as apply_wavlm_patch
    apply_wavlm_patch(seg)

    # 3-D input (batch, channel, sample) per DiariZen's forward.
    example = torch.zeros(1, mods.export_channels, mods.seg_window_samples, dtype=torch.float32)

    # Trace, then convert to an ML Program. Prefer FIXED shapes (fastest, most
    # ANE-friendly); use ct.EnumeratedShapes only if a few chunk sizes are needed;
    # avoid fully dynamic shapes.
    traced = torch.jit.trace(seg, example)

    compute_precision = (
        ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="waveform", shape=example.shape)],
        compute_precision=compute_precision,
        # Let Swift pick compute units at load time; converting for `ALL` keeps
        # the ANE path available. Profiling in Phase 4 decides the runtime default.
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"DiariZen segmentation ({args.model}) — CC BY-NC weights"
    seg_path = os.path.join(args.out, "Segmentation.mlpackage")
    mlmodel.save(seg_path)
    print(f"wrote {seg_path}")

    if mods.embedding is not None:
        print("embedding model present — TODO: convert in phase 2")

    print(
        "\nNext: `python validation/compare_pytorch_coreml.py --model",
        args.model,
        "--coreml",
        args.out,
        "` (expect looser tol than ONNX because of FP16).",
    )


if __name__ == "__main__":
    main()
