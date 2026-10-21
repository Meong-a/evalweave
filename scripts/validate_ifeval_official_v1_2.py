#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from group_llm_benchmark.tasks.factory import create_task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "configs/tasks_v1_2/ifeval_full_v1_2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = create_task(config, base_dir=config_path.parent)
    samples = task.load_samples()
    if len(samples) != 541:
        raise ValueError(f"IFEval requires 541 samples, got {len(samples)}")

    metric_names = {
        "strict_prompt_accuracy",
        "loose_prompt_accuracy",
        "strict_instruction_accuracy",
        "loose_instruction_accuracy",
    }
    checked = 0
    instruction_checks = 0
    for sample in samples:
        # This is deliberately not a benchmark prediction. It only exercises every
        # official checker against a deterministic non-empty compatibility response.
        metrics = task.score("Compatibility validation response.", sample)
        if set(metrics) != metric_names:
            raise ValueError(f"{sample.sample_id}: unexpected metrics {sorted(metrics)}")
        if any(value < 0.0 or value > 1.0 for value in metrics.values()):
            raise ValueError(f"{sample.sample_id}: metric outside [0, 1]")
        checked += 1
        instruction_checks += len(sample.reference["instruction_id_list"])

    report = {
        "status": "PASS",
        "suite_version": "1.2.2",
        "validation": "official_ifeval_scorer_compatibility_only",
        "not_a_model_benchmark_result": True,
        "samples_checked": checked,
        "instruction_checks_exercised": instruction_checks,
        "official_metrics_returned": sorted(metric_names),
    }
    destination = root / "manifests/ifeval_official_validation_v1_2.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
