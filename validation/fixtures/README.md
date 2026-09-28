Golden fixtures for parity tests live here. They are **git-ignored**
(`*.wav`, `*.npy`, `*.npz`, `*.rttm`) because they are audio or model-derived
outputs (the checkpoints are CC-BY-NC) and are cheap to regenerate. Keep them
SMALL and at the target sample rate (16 kHz mono) so parity tests the model, not
the resampler.

## `parity_16k.wav`  (tensor-parity fixture)
30 s / 16 kHz / mono clip (AMI `EN2002a_30s`, multi-speaker). Used by
`compare_pytorch_onnx.py` and `compare_pytorch_coreml.py` — the first 16 s window
(256000 samples) is one segmentation chunk. Regenerate from the DiariZen sample:

    sox build/DiariZen/example/EN2002a_30s.wav -r 16000 -c 1 \
        validation/fixtures/parity_16k.wav

## `EN2002a.golden.rttm`  (end-to-end reference)
The upstream DiariZen RTTM for `EN2002a_30s.wav` (3 speakers, with overlap),
produced by the pure-PyTorch pipeline. The Phase 1 end-to-end check
(`reproduce_rttm.py`) reproduces it byte-for-byte with ONNX segmentation
(DER 0.000%), and the Swift port will diff against it. Regenerate:

    python validation/reproduce_rttm.py   # writes this file, then asserts DER≈0
