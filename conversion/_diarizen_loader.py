"""Shared helpers to load a DiariZen pipeline and pull out the exportable
neural sub-modules.

Phase 0 findings (from upstream source — see docs/PIPELINE.md):

    DiariZenPipeline  (subclasses pyannote SpeakerDiarization)
      ├── segmentation model  =  pipeline._segmentation.model
      │     pyannote Model wrapping WavLM(+pruning) + Conformer, POWERSET head.
      │     forward(waveforms: (batch, channel, sample)) -> (batch, frame, classes)
      │     (`assert waveforms.dim() == 3`; it selects one channel internally).
      │     THIS is the model we must convert. Fixed chunk length = static shapes.
      ├── embedding model     =  pyannote/wespeaker-voxceleb-resnet34-LM
      │     WeSpeaker ResNet34 — already ported to CoreML by FluidInference, so
      │     the embedding stage can largely be reused rather than re-converted.
      └── clustering          =  VBx (+ PLDA/LDA) or Agglomerative, CPU (pyannote).

The heavy import is done lazily so this file lints without the DiariZen deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiariZenModules:
    segmentation: object          # pyannote Model: WavLM + Conformer, powerset head
    embedding: object | None      # WeSpeaker ResNet34 (reusable CoreML port exists)
    sample_rate: int              # 16000
    num_channels: int             # channels the Model *declares* (8 for base-s80-md)
    export_channels: int          # channels to trace/export with = 1 (mono; picks ch 0)
    seg_window_samples: int       # chunk length in samples = seg_duration * sample_rate
    powerset_classes: int         # segmentation output classes (per frame)


def load_pipeline(model_id: str, cache_dir: str | None = None):
    """Load the upstream DiariZen pipeline (downloads the CC-BY-NC checkpoint)."""
    from diarizen.pipelines.inference import DiariZenPipeline  # type: ignore

    return DiariZenPipeline.from_pretrained(model_id, cache_dir=cache_dir)


def extract_modules(pipeline) -> DiariZenModules:
    """Locate the exportable segmentation nn.Module + its input geometry."""
    seg = pipeline._segmentation.model
    seg.eval()

    sample_rate = int(getattr(seg, "sample_rate", 16000))
    num_channels = int(getattr(seg, "num_channels", 1))

    # Chunk length: pyannote exposes the segmentation window as a duration (s).
    # Prefer the pipeline's configured value; fall back to the model's chunk_size.
    seg_duration = (
        getattr(getattr(pipeline, "_segmentation", None), "duration", None)
        or getattr(seg, "chunk_size", None)
    )
    if seg_duration is None:
        raise RuntimeError(
            "could not determine segmentation chunk duration — inspect "
            "pipeline._segmentation and seg.chunk_size"
        )
    seg_window_samples = int(round(float(seg_duration) * sample_rate))

    # Powerset class count from the model's specifications.
    spec = seg.specifications
    powerset_classes = int(getattr(spec, "num_powerset_classes", 0)) or -1

    return DiariZenModules(
        segmentation=seg,
        embedding=None,  # WeSpeaker — reuse existing CoreML port; convert later if needed
        sample_rate=sample_rate,
        num_channels=num_channels,
        export_channels=1,  # mono inference; the model selects channel 0
        seg_window_samples=seg_window_samples,
        powerset_classes=powerset_classes,
    )
