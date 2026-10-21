#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from group_llm_benchmark.tasks.factory import create_task


EXPECTED = {
    "mmlu_test_full_v1_2.json": 14042,
    "ceval_val_full_v1_2.json": 1346,
    "ceval_test_submission_full_v1_2.json": 12342,
    "bbh_full_cot_v1_2.json": 6511,
    "gsm8k_test_full_v1_2.json": 1319,
    "truthfulqa_mc1_full_v1_2.json": 790,
    "ifeval_full_v1_2.json": 541,
    "humaneval_full_v1_2.json": 164,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_root = root / "configs/tasks_v1_2"
    report: dict[str, object] = {
        "status": "PASS",
        "suite_version": "1.2.2",
        "validation": "full_task_plugin_load_and_request_build",
        "tasks": {},
    }

    for filename, expected_count in EXPECTED.items():
        config_path = config_root / filename
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task = create_task(config, base_dir=config_path.parent)
        samples = task.load_samples()
        if len(samples) != expected_count:
            raise ValueError(f"{filename}: expected {expected_count}, got {len(samples)}")
        first_request = task.build_request(samples[0])
        last_request = task.build_request(samples[-1])
        if not first_request.prompt.strip() or not last_request.prompt.strip():
            raise ValueError(f"{filename}: empty generated prompt")

        task_report = {
            "sample_count": len(samples),
            "first_sample_id": samples[0].sample_id,
            "last_sample_id": samples[-1].sample_id,
            "first_prompt_characters": len(first_request.prompt),
            "last_prompt_characters": len(last_request.prompt),
        }
        report["tasks"][filename] = task_report  # type: ignore[index]
        print(f"validated {filename}: {len(samples)}")

    destination = root / "manifests/task_plugin_validation_v1_2.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
