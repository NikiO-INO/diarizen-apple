# Upstream export patches

If tracing/export needs monkey-patches to DiariZen or its pyannote fork
(e.g. replacing an un-traceable op), put minimal patch files here and apply them
from the conversion scripts. Keep them small and documented.

## `wavlm_export_patch.py` — applied by `export_coreml.py`

Rewrites two lines of `Wav2Vec2Model.extract_features` so coremltools can trace it.
Both are behavioural no-ops for our export (fixed mono 16 s chunk, `lengths=None`,
eval mode); the ONNX/CoreML parity checks confirm outputs are unchanged.

1. **Un-scriptable variable-length `layer_norm`.** `extract_features_mc` is
   `@torch.jit.export`, which forces torch to *script* the whole module during
   `torch.jit.trace`; scripting then fails on
   `F.layer_norm(wave[:length], (length,))` (normalized_shape is `Tuple[Tensor]`,
   not `List[int]`). That branch only runs when `lengths is not None`.
2. **`GradMultiply` → `pythonop`.** It's a `torch.autograd.Function` whose forward
   is identity (`x.new(x)`); it only scales gradients in backward. coremltools
   can't convert the resulting `pythonop`, so we drop it for inference.

ONNX export doesn't need this patch (`torch.onnx.export` takes a different path).
