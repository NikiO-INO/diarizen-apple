"""Which compute device does CoreML assign each op to? Answers "does the ANE get
used?" by loading the MLComputePlan and tallying the preferred device per operation.

    python benchmarks/compute_plan.py [ne|gpu|all]   # default: ne

Finding for diarizen-wavlm-base-s80-md segmentation: under CPU+GPU every op prefers
the GPU; under CPU+ANE every op prefers the CPU (0 ops on the ANE). The WavLM+
Conformer transformer (linear/matmul/layer_norm/reshape/transpose-heavy) is not in
an ANE-friendly form, so CoreML's planner keeps it off the Neural Engine — which is
why `.cpuAndGPU` is the default and we don't claim "runs on the ANE".
"""

from __future__ import annotations

import sys
from collections import Counter


def main() -> None:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    policy = {
        "ne": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[sys.argv[1] if len(sys.argv) > 1 else "ne"]
    pkg = sys.argv[2] if len(sys.argv) > 2 else "build/coreml/Segmentation.mlpackage"

    # MLComputePlan needs a compiled .mlmodelc; keep the MLModel alive so the temp
    # compiled dir isn't GC-deleted before the plan loads.
    model = ct.models.MLModel(pkg, compute_units=policy)
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=policy)
    prog = plan.model_structure.program
    assert prog is not None, "not an ML Program"

    def dev(d):
        n = type(d).__name__
        return "ANE" if "Neural" in n else "GPU" if "GPU" in n else "CPU" if "CPU" in n else n

    by_device = Counter()
    not_ane = Counter()
    for func in prog.functions.values():
        for op in func.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            by_device[dev(usage.preferred_compute_device)] += 1
            if not any("Neural" in type(d).__name__ for d in usage.supported_compute_devices):
                not_ane[op.operator_name] += 1

    total = sum(by_device.values())
    print(f"policy={policy}  total ops={total}")
    for d, c in by_device.most_common():
        print(f"  {d:4s}: {c:4d}  ({100 * c / total:.1f}%)")
    if not_ane:
        print("op types NOT ANE-supported:", dict(not_ane.most_common(12)))


if __name__ == "__main__":
    main()
