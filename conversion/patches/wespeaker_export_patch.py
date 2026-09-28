"""Export-time monkey-patch for pyannote's `StatsPool` (WeSpeaker embedding).

Problem: `StatsPool.forward` computes masked statistics with
`torch.vmap(self._pool, in_dims=(None, 1))(sequences, weights)` over a speaker
dimension. `torch.vmap` cannot be traced/converted (it also emits `movedim`,
which coremltools has no lowering for).

Our embedding export always runs the single-speaker path (weights shaped
`(batch, frames)`), for which the vmap collapses to a single `_pool` call. We
replace `forward` with that direct, traceable equivalent — same math, no vmap,
no movedim, no einops rearrange. Parity tests confirm the embedding is unchanged.
"""

from __future__ import annotations


def apply() -> None:
    import torch
    import torch.nn.functional as F

    from pyannote.audio.models.blocks.pooling import StatsPool

    def forward(self, sequences, weights=None):
        # sequences: (batch, channels, frames)
        if weights is None:
            mean = sequences.mean(dim=-1)
            std = sequences.std(dim=-1, correction=1)
            return torch.cat([mean, std], dim=-1)

        # Single-speaker weights: (batch, frames) — or (batch, 1, frames).
        w = weights
        if w.dim() == 3:
            w = w[:, 0, :]
        w = w.unsqueeze(1)  # (batch, 1, frames)

        num_frames = sequences.shape[-1]
        if w.shape[-1] != num_frames:
            w = F.interpolate(w, size=num_frames, mode="nearest")

        # _pool expects (batch, frames); it re-adds the channel dim internally.
        return self._pool(sequences, w.squeeze(1))

    StatsPool.forward = forward
