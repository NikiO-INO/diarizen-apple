"""Shared numerical-parity helpers for the reference/production backends.

Parity is the backbone of this port: every backend must reproduce the PyTorch
reference within an explicit tolerance on fixed golden inputs. Keep tolerances
here so ONNX and CoreML checks agree on what "close enough" means.
"""

from __future__ import annotations

from dataclasses import dataclass


# Tolerances.
#
# ONNX runs FP32 on CPU and reproduces PyTorch almost exactly — hold it tight.
ONNX_ATOL, ONNX_RTOL = 1e-3, 1e-3
#
# CoreML is compared in PROBABILITY space, not on the raw log-softmax logits, and
# by decision agreement — see compare_logprob_segmentation() for why. The
# segmentation head emits log-probs reaching ~-14; the CoreML runtime (Espresso/
# BNNS) diverges from PyTorch by ~0.02-0.2 *in that deep-negative tail*, which is
# pure numerical accumulation over 12 transformer layers (present in FP32 too, so
# it is NOT FP16 rounding and NOT a conversion bug — the graph is structurally
# faithful, proven by the 1e-5 ONNX match). In probability space that tail maps to
# ~0, so it is downstream-irrelevant: measured prob-space error is <1e-3 mean /
# ~2e-2 max and the winning powerset class agrees on >99.7% of frames.
COREML_PROB_ATOL = 4e-2          # max |Δprob|; observed ~2.1e-2, ~2x margin
COREML_ARGMAX_MIN_AGREE = 0.99   # fraction of frames whose argmax powerset class matches
# Legacy raw-logit tolerance, kept only for ad-hoc FP32 spot checks. Do NOT gate
# the CoreML segmentation parity on this — the log-softmax tail makes max_abs a
# misleading proxy for accuracy (a 0.2 delta at logit -14 changes no decision).
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


@dataclass
class SegParityResult:
    name: str
    prob_max_abs: float
    prob_mean_abs: float
    argmax_agree: float      # fraction in [0, 1]
    logit_max_abs: float     # raw log-prob delta, reported for transparency
    passed: bool

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"[{flag}] {self.name}: prob_max={self.prob_max_abs:.3e} "
            f"prob_mean={self.prob_mean_abs:.3e} argmax_agree={self.argmax_agree * 100:.2f}% "
            f"(raw logit_max={self.logit_max_abs:.3e})"
        )


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


def compare_logprob_segmentation(
    name: str,
    reference,
    candidate,
    prob_atol: float = COREML_PROB_ATOL,
    min_argmax_agree: float = COREML_ARGMAX_MIN_AGREE,
) -> SegParityResult:
    """Decision-level parity for the log-softmax powerset segmentation output.

    Inputs are log-probabilities shaped (..., frame, class). We compare in
    probability space (exp) and by per-frame argmax agreement, because the raw
    logits reach ~-14 where the CoreML runtime's accumulated numerical error is
    largest yet irrelevant — exp() maps it to ~0 and it changes no decision.
    `passed` requires BOTH the probability-space max delta and the argmax
    agreement to clear their thresholds.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        return SegParityResult(
            f"{name} (shape {ref.shape} vs {cand.shape})",
            float("inf"), float("inf"), 0.0, float("inf"), False,
        )
    logit_max = float(np.abs(ref - cand).max())
    ref_p, cand_p = np.exp(ref), np.exp(cand)
    prob_diff = np.abs(ref_p - cand_p)
    prob_max, prob_mean = float(prob_diff.max()), float(prob_diff.mean())
    agree = float((ref.argmax(-1) == cand.argmax(-1)).mean())
    passed = prob_max <= prob_atol and agree >= min_argmax_agree
    return SegParityResult(name, prob_max, prob_mean, agree, logit_max, passed)


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
