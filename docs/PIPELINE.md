# DiariZen pipeline map (Phase 0)

Derived from upstream source (`build/DiariZen`, `diarizen-wavlm-*-s80-md`).

## Structure

`DiariZenPipeline` **subclasses pyannote's `SpeakerDiarization` pipeline** and
plugs in DiariZen's segmentation model + a WeSpeaker embedder + VBx clustering.
So most orchestration (sliding window, binarize, embedding over active regions,
clustering, stitching) is **pyannote's**, and only the segmentation network is
DiariZen-specific.

```
audio (16 kHz mono)
  → [SEGMENTATION]  DiariZen WavLM(+pruning) + Conformer, POWERSET head   ← convert
        pipeline._segmentation.model
        forward(waveforms: (batch, channel, sample)) -> (batch, frame, classes)
        fixed chunk = seg_duration * 16000 samples  (static shapes)
  → [BINARIZE + median filter]  (pyannote / scipy)                        ← Swift/CPU
  → [EMBEDDING]  pyannote/wespeaker-voxceleb-resnet34-LM (WeSpeaker R34)  ← reuse CoreML port
        embedding_exclude_overlap=True
  → [CLUSTERING]  VBx (+ PLDA/LDA) or AgglomerativeClustering            ← Swift/CPU (pyannote)
        config.toml: ahc_criterion, ahc_threshold, Fa, Fb, lda_dim, max_iters,
                     min/max_speakers, min_cluster_size, plda/ dir
  → RTTM
```

## What we convert vs reuse

| Stage | Source | Plan |
|---|---|---|
| Segmentation (WavLM+Conformer, powerset) | DiariZen `pytorch_model.bin` | **Convert** to ONNX (ref) + CoreML (prod) |
| Embedding (WeSpeaker ResNet34) | `pyannote/wespeaker-voxceleb-resnet34-LM` | **Reuse** — FluidInference already ships a CoreML WeSpeaker; else convert |
| Clustering (VBx + PLDA) | pyannote + DiariZen config | **Port to Swift/CPU** (or reuse existing VBx) |
| Sliding window / binarize / stitch | pyannote | **Reimplement in Swift** |

## Segmentation forward (the export target)

- Input: `(batch, channel, sample)`, 3-D (`assert waveforms.dim() == 3`); it
  selects `self.selected_channel` internally. Single-channel checkpoint →
  `num_channels == 1`, feed `(1, 1, seg_window_samples)`.
- Body: `wav2wavlm` (WavLM features) → `weight_sum` (layer weights) → `proj` →
  `lnorm` → `conformer` → `classifier` → `activation`.
- Output: `(batch, frame, classes)` powerset scores.
- `sample_rate = 16000`.

## Confirmed live — `diarizen-wavlm-base-s80-md`

Loaded via `DiariZenPipeline.from_pretrained(...)` + a forward pass:

| Field | Value |
|---|---|
| segmentation model | `diarizen.models.eend.model_wavlm_conformer.Model` |
| WavLM | `wavlm_base_s80_md`, 13 layers, feat_dim 768 |
| Conformer | attention_in 256, ffn_hidden 1024, num_head 4, num_layer 4 |
| `seg_duration` | **16 s** → `seg_window_samples = 256000` |
| segmentation_step | 0.1 s |
| declared `num_channels` | 8, but `selected_channel = 0` — **feed mono `(1, 1, 256000)`** |
| output | **`(1, 799, 11)`** = (batch, frames, powerset_classes); `num_frames = 799` |
| clustering | **VBxClustering**: ahc_threshold 0.6, Fa 0.07, Fb 0.8, lda_dim 128, max_iters 20, min/max_speakers 1/20 (+ PLDA/LDA dir) |
| embedding | `pyannote/wespeaker-voxceleb-resnet34-LM` (WeSpeaker ResNet34) |

- ✅ **The WavLM + Conformer forward traces cleanly in eval on a fixed 16 s chunk**
  — good omen for `torch.onnx.export` / `coremltools` (static shapes).
- Env that loads it: Python 3.11 + torch 2.1.1 + torchaudio 2.1.1 + accelerate
  0.29 (NOT transformers 5.x — it forces torch ≥ 2.5) + the DiariZen pyannote
  fork. See ROADMAP Phase 0.

### Export input note
Even though the model declares `num_channels = 8`, inference feeds **mono** and it
picks `selected_channel = 0`. Export/trace with `(1, 1, 256000)`.

## Export-relevant risks

- **WavLM tracing**: transformer self-attention can trip torch.onnx / coremltools.
  Fixed chunk length removes dynamic-shape issues; watch for masked-softmax /
  `where` ops needing opset ≥ 17 or a small patch (`conversion/patches/`).
- **Structured pruning** (`hardconcrete`, `pruning_utils`) is baked into the
  weights at inference — should be a plain forward, but verify no train-only
  masking path is hit in eval().
