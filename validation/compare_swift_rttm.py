"""End-to-end Phase-3 gate: the Swift diarizen-cli RTTM vs the pyannote oracle RTTM.

Runs the full Swift pipeline (CoreML seg → Swift fbank + CoreML embedding →
Swift VBx clustering → reconstruction) on the byte-identical waveform the oracle
saw, then measures DER against build/oracle/EN2002a.coreml.rttm (pyannote with the
same CoreML segmentation). Target: DER ≈ 0.

    swift build
    python validation/dump_rttm_oracle.py                 # writes the oracle RTTM
    python validation/compare_swift_rttm.py \
        --cli .build/debug/diarizen-cli \
        --coreml build/coreml \
        --wav build/DiariZen/example/EN2002a_30s.wav \
        --oracle build/oracle/EN2002a.coreml.rttm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

DER_TOL = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default=".build/debug/diarizen-cli")
    ap.add_argument("--coreml", default="build/coreml")
    ap.add_argument("--wav", default="build/DiariZen/example/EN2002a_30s.wav")
    ap.add_argument("--oracle", default="build/oracle/EN2002a.coreml.rttm")
    args = ap.parse_args()

    import numpy as np
    import torchaudio
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    if not os.path.exists(args.cli):
        print(f"CLI not found at {args.cli} — run `swift build` first.", file=sys.stderr)
        return 2

    waveform, _ = torchaudio.load(args.wav)
    samples = np.ascontiguousarray(waveform[0].numpy(), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "audio.f32")
        swift_rttm = os.path.join(tmp, "swift.rttm")
        samples.tofile(raw)
        subprocess.run(
            [args.cli, "--models", args.coreml, "--compute-units", "cpu",
             "--raw-input", raw, "--output", swift_rttm],
            check=True,
        )
        with open(swift_rttm) as f:
            swift_text = f.read()
        # normalize the uri to match the oracle so the metric aligns the file
        swift_fixed = os.path.join(tmp, "swift_fixed.rttm")
        with open(swift_fixed, "w") as f:
            f.write("\n".join(
                parts_with_uri(line, "EN2002a") for line in swift_text.splitlines() if line.strip()
            ) + "\n")
        hyp = next(iter(load_rttm(swift_fixed).values()))

    ref = next(iter(load_rttm(args.oracle).values()))
    metric = DiarizationErrorRate()
    der = metric(ref, hyp)

    passed = der <= DER_TOL
    print(f"[{'PASS' if passed else 'FAIL'}] end-to-end Swift RTTM: DER={der*100:.3f}% (tol {DER_TOL*100:.0f}%)")
    print(f"    ref turns={len(list(ref.itertracks()))}  swift turns={len(list(hyp.itertracks()))}")
    if not passed:
        comp = metric[:]
        print(f"    components: {comp}")
    return 0 if passed else 1


def parts_with_uri(line: str, uri: str) -> str:
    p = line.split()
    if len(p) >= 2:
        p[1] = uri
    return " ".join(p)


if __name__ == "__main__":
    raise SystemExit(main())
