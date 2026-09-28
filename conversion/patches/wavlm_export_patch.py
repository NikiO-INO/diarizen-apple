"""Export-time monkey-patch for DiariZen's WavLM (`Wav2Vec2Model`).

Problem: `Wav2Vec2Model.extract_features_mc` is decorated `@torch.jit.export`,
which forces torch.jit to *script* the whole module when coremltools traces it.
Scripting then chokes on the variable-length path in both `extract_features` /
`extract_features_mc`:

    F.layer_norm(wave[:length], (length,))   # normalized_shape is Tuple[Tensor]

That path only runs when `lengths is not None`. We always export a single fixed
16 s chunk with `lengths=None`, so we replace both methods with the equivalent
`lengths is None` version (plain, traceable) and drop the jit-export marker.
Behaviourally identical for our inputs; ONNX export already validated the
numbers, so parity tests will confirm this patch changes nothing.
"""

from __future__ import annotations


def apply(model) -> None:
    """Patch every Wav2Vec2Model instance/class reachable from `model`."""
    import torch.nn.functional as F

    from diarizen.models.module.wav2vec2.model import Wav2Vec2Model

    def extract_features(self, waveforms, lengths=None, num_layers=None):
        if self.normalize_waveform:
            waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])
        x, lengths = self.feature_extractor(waveforms, lengths)
        # GradMultiply is a gradient-scaling identity — forward returns a copy of
        # x and only scales gradients in backward. At inference it is a no-op, and
        # as a torch.autograd.Function it traces to a `pythonop` that coremltools
        # cannot convert. Drop it entirely for export.
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    # Replace both methods with the plain (lengths=None) version. Assigning a
    # fresh function also drops the `@torch.jit.export` marker, so tracing no
    # longer force-scripts the module.
    Wav2Vec2Model.extract_features = extract_features
    Wav2Vec2Model.extract_features_mc = extract_features
