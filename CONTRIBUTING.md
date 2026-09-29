# Contributing

Thanks for looking at `diarizen-apple`. This is a faithful port of the DiariZen
pipeline to Apple Silicon, so the guiding rule is that changes keep the output
matching the original within the recorded tolerances.

## Getting set up

The runtime is pure Swift and needs only Xcode.

```bash
swift build -c release
swift test
```

Tests are pure Swift and CoreML-decode logic, so they run without any model
files. Please build in release at least once before sending a change: a few bugs
here only appear under the optimizer (the powerset combination code once
miscompiled under `-O`), and CI builds release for that reason.

The Python side is only for exporting models and running parity checks.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are CC BY-NC 4.0 and are never committed. Download a checkpoint
yourself (see `MODELS.md`) and export it locally before running the checks.

## Running the checks

The parity harness in `validation/` asserts three levels, from the frontend to
the full pipeline:

```bash
# Swift filterbank against Kaldi
python validation/compare_swift_fbank.py --model BUT-FIT/diarizen-wavlm-base-s80-md

# CoreML segmentation and embeddings against PyTorch
python validation/compare_pytorch_coreml.py     --model BUT-FIT/diarizen-wavlm-base-s80-md --coreml build/coreml
python validation/compare_pytorch_embedding.py  --model BUT-FIT/diarizen-wavlm-base-s80-md --coreml build/coreml

# Full Swift pipeline against the upstream RTTM, and DER on a real meeting
python validation/compare_swift_rttm.py --model BUT-FIT/diarizen-wavlm-base-s80-md --coreml build/coreml
python validation/eval_ami.py           --model BUT-FIT/diarizen-wavlm-base-s80-md --coreml build/coreml
```

Benchmarks live in `benchmarks/`:

```bash
benchmarks/run.sh              # RTF / latency / peak memory, Swift vs PyTorch
python benchmarks/compute_plan.py --coreml build/coreml   # per-op device placement
```

If a change can affect accuracy, include the parity and DER numbers before and
after in the pull request.

## Conventions

- Neural stages run in CoreML. The frontend, aggregation, clustering, and
  reconstruction stay in Swift on the CPU. Keep that split so each stage stays
  testable against the original.
- Neural outputs are checked with cosine or numerical tolerances. Deterministic
  stages (decode, clustering math, RTTM) are held bit-exact against an oracle.
- DiariZen owns the frontend semantics. When Swift and a reference disagree,
  match DiariZen, and record the tolerance.
- Do not commit weights or converted artifacts (`.mlpackage`, `.onnx`, `.pt`,
  fixtures). They are git-ignored and non-redistributable.

## Where help is welcome

- **Neural Engine support.** The segmentation model does not run on the ANE today
  (see the README limitations). Getting it there means an ANE-principles rewrite
  of the WavLM and Conformer, not a conversion flag.
- **An MLX backend.** A re-implementation with mapped weights, revalidated
  against the CoreML and PyTorch outputs.
- **large-s80-md-v2 tensor fidelity.** The large model's CoreML segmentation
  diverges from PyTorch (tolerated downstream, but worth reducing).
- **More validation audio.** Longer files with reliable references would firm up
  the accuracy and real-time-factor numbers.

## Pull requests

Keep changes focused and tested. Run `swift test`, and run the relevant parity
check when the change touches a stage. By contributing you agree that your code
is under the repository's MIT license.
