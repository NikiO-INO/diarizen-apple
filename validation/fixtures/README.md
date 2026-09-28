Golden fixtures for parity tests live here (git-ignored: *.wav/*.npy/*.npz).

Generate with Phase 0/1:
  - a few short test clips (e.g. 30s, 2 speakers) at 16 kHz mono
  - the PyTorch reference tensors for each (segmentation output)
Keep them SMALL and at the target sample rate so parity tests the model, not the
resampler.
