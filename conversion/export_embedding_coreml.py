"""Export the WeSpeaker ResNet34 embedding *neural* part to native CoreML.

Per the architecture decision, the deterministic kaldi fbank frontend stays in
native Swift (Accelerate/vDSP); ONLY the learned ResNet34 + masked statistics
pooling go to CoreML. So this converts a wrapper whose input is the ALREADY
computed (CMVN'd) fbank, not a waveform:

    forward(fbank (1, F, 80), weights (1, W)) -> embedding (1, 256)

where F = kaldi frames for a 16 s / 256000-sample crop (the pipeline's embedding
window == the segmentation window) and W = segmentation frames (799); the pooling
linearly interpolates `weights` to the conv output length internally.

    python conversion/export_embedding_coreml.py \
        --model BUT-FIT/diarizen-wavlm-base-s80-md \
        --out build/coreml \
        --precision fp16

Phase 2 (embedding stage). Fbank is Swift's job — see the parity scripts.
"""

from __future__ import annotations

import argparse
import os

# Embedding crop == segmentation window (16 s); segmentation emits 799 frames.
CROP_SAMPLES = 256_000
SEG_FRAMES = 799


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    ap.add_argument("--out", default="build/coreml")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    import coremltools as ct
    import torch

    from _diarizen_loader import load_pipeline

    from patches.wespeaker_export_patch import apply as apply_statspool_patch

    os.makedirs(args.out, exist_ok=True)
    pipe = load_pipeline(args.model)
    wespeaker = pipe._embedding.model_.eval()

    # StatsPool.forward uses torch.vmap (un-traceable); swap in the single-speaker
    # equivalent before tracing. No-op numerically — see the patch + parity check.
    apply_statspool_patch()

    # The neural part only: (CMVN'd fbank, weights) -> 256-d embedding.
    class ResNetEmbedding(torch.nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet

        def forward(self, fbank, weights):
            return self.resnet(fbank, weights)[1]  # embed_b (LM head)

    net = ResNetEmbedding(wespeaker.resnet).eval()

    # Derive the real fbank frame count F from the actual frontend on a 16 s crop.
    with torch.no_grad():
        fbank = wespeaker.compute_fbank(torch.zeros(1, 1, CROP_SAMPLES))
    n_frames = int(fbank.shape[1])
    n_mel = int(fbank.shape[2])
    print(f"fbank frames F={n_frames}, mel={n_mel}, weights W={SEG_FRAMES}")

    ex_fbank = torch.zeros(1, n_frames, n_mel, dtype=torch.float32)
    ex_weights = torch.ones(1, SEG_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(net, (ex_fbank, ex_weights))

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="fbank", shape=ex_fbank.shape),
            ct.TensorType(name="weights", shape=ex_weights.shape),
        ],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"DiariZen WeSpeaker ResNet34 embedding ({args.model}) — CC BY-NC weights"
    out_path = os.path.join(args.out, "Embedding.mlpackage")
    mlmodel.save(out_path)
    print(f"wrote {out_path}")
    print("\nNext: `python validation/compare_pytorch_embedding.py` (PyTorch vs CoreML, cosine).")


if __name__ == "__main__":
    main()
