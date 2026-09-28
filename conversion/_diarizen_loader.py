"""Shared helpers to load a DiariZen pipeline and pull out the exportable
neural sub-modules.

Phase 0 deliverable: fill in the TODOs by inspecting the actual objects returned
by `DiariZenPipeline.from_pretrained(...)`. The upstream package is required
(`pip install -e DiariZen`). This module deliberately does NOT import DiariZen at
top level so the file is importable for linting without the heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiariZenModules:
    """The neural pieces we convert, plus the shapes needed to trace them."""

    segmentation: object          # nn.Module: WavLM + Conformer + powerset head
    embedding: object | None      # nn.Module: speaker-embedding extractor (may be fused)
    sample_rate: int              # expected input sample rate (16000)
    seg_window_samples: int       # segmentation chunk length in samples (e.g. 10s * 16000)
    # TODO(phase0): record the exact forward() signature + output shape of each.


def load_pipeline(model_id: str):
    """Load the upstream DiariZen pipeline (downloads the CC-BY-NC checkpoint)."""
    from diarizen.pipelines.inference import DiariZenPipeline  # type: ignore

    return DiariZenPipeline.from_pretrained(model_id)


def extract_modules(pipeline) -> DiariZenModules:
    """Locate the exportable nn.Modules inside the pipeline.

    TODO(phase0): DiariZen wraps a segmentation model (WavLM+Conformer, powerset
    output) and clustering. Inspect `pipeline`:
        - the segmentation model is usually under something like
          `pipeline._segmentation.model` / `pipeline.segmentation_model`;
        - the embedding extractor may be a separate model or fused into
          segmentation — confirm and set `embedding=None` if fused.
    Put the eval()-mode modules and their input geometry into DiariZenModules.
    """
    raise NotImplementedError(
        "Phase 0: inspect the DiariZen pipeline object and return DiariZenModules. "
        "Run `python -c 'from diarizen.pipelines.inference import DiariZenPipeline; "
        "p=DiariZenPipeline.from_pretrained(\"BUT-FIT/diarizen-wavlm-base-s80-md\"); "
        "print(type(p)); print([a for a in dir(p) if not a.startswith(\"__\")])'` "
        "and fill this in."
    )
