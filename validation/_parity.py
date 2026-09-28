"""Shared numerical-parity helpers for the reference/production backends.

Parity is the backbone of this port: every backend must reproduce the PyTorch
reference within an explicit tolerance on fixed golden inputs. Keep tolerances
here so ONNX and CoreML checks agree on what "close enough" means.
"""

from __future__ import annotations

from dataclasses import dataclass


# Tolerances. ONNX (FP32) should be tight; CoreML (FP16) needs looser bounds.
ONNX_ATOL, ONNX_RTOL = 1e-3, 1e-3
COREML_ATOL, COREML_RTOL = 2e-2, 2e-2


@dataclass
class ParityResult:
    name: str
    max_abs: float
    mean_abs: float
    passed: bool

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: max_abs={self.max_abs:.3e} mean_abs={self.mean_abs:.3e}"


def compare(name: str, reference, candidate, atol: float, rtol: float) -> ParityResult:
    """Compare two array-likes; `passed` uses numpy.allclose semantics."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    if ref.shape != cand.shape:
        return ParityResult(f"{name} (shape {ref.shape} vs {cand.shape})", float("inf"), float("inf"), False)
    diff = np.abs(ref - cand)
    passed = bool(np.allclose(ref, cand, atol=atol, rtol=rtol))
    return ParityResult(name, float(diff.max()), float(diff.mean()), passed)


def load_fixture_audio(path: str, sample_rate: int):
    """Load a mono waveform at `sample_rate` as float32 in [-1, 1]."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(
            f"fixture {path} is {sr} Hz; resample to {sample_rate} first "
            "(keep fixtures at the target rate so parity isn't testing the resampler)."
        )
    return np.ascontiguousarray(audio, dtype=np.float32)
