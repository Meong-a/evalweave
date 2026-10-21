#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TASKS = {
    "mmlu": ("mmlu_test_full_v1_2.json", 16),
    "ceval_val": ("ceval_val_full_v1_2.json", 16),
    "bbh": ("bbh_full_cot_v1_2.json", 512),
    "gsm8k": ("gsm8k_test_full_v1_2.json", 512),
    "truthfulqa": ("truthfulqa_mc1_full_v1_2.json", 16),
    "ifeval": ("ifeval_full_v1_2.json", 1024),
    "ceval_test_submission": ("ceval_test_submission_full_v1_2.json", 16),
    "humaneval_generation": ("humaneval_full_v1_2.json", 512),
}
DIRECT_SCORE_TASKS = ("mmlu", "ceval_val", "bbh", "gsm8k", "truthfulqa", "ifeval")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "model"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create full EvalWeave v1.2.1 experiment configs")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    root = args.root.resolve()
    model_config = args.model_config.resolve()
    output_dir = args.output_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    model = json.loads(model_config.read_text(encoding="utf-8"))
    model_id = safe_name(str(model.get("id") or model.get("model") or model_config.stem))
    experiment_paths: dict[str, Path] = {}

    for task_key, (task_filename, max_new_tokens) in TASKS.items():
        destination = output_dir / f"{model_id}_{task_key}_full_v1_2_1.json"
        payload = {
            "schema_version": "1.0",
            "experiment_name": f"{model_id}_{task_key}_full_v1_2_1",
            "model_ref": str(model_config),
            "task_ref": str(root / "configs/tasks_v1_2" / task_filename),
            "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "batch_size": args.batch_size,
            "seed": 42,
            "resume": True,
            "output_root": str(runs_dir),
            "metadata": {"suite_version": "1.2.1", "scope": "full"},
        }
        write_json(destination, payload)
        experiment_paths[task_key] = destination

    suite_path = output_dir / f"{model_id}_direct_scored_suite_full_v1_2_1.json"
    write_json(
        suite_path,
        {
            "schema_version": "1.0",
            "suite_name": f"{model_id}_direct_scored_suite_full_v1_2_1",
            "mode": "matrix",
            "experiments": [str(experiment_paths[name]) for name in DIRECT_SCORE_TASKS],
            "output_root": str(root / "suite_reports"),
            "metadata": {
                "scope": "full",
                "note": "HumanEval pass@1 and C-Eval test submission are postprocessed separately.",
            },
        },
    )
    manifest = {
        "suite_version": "1.2.1",
        "model_id": model_id,
        "model_config": str(model_config),
        "full_experiments": {name: str(path) for name, path in experiment_paths.items()},
        "direct_scored_suite": str(suite_path),
    }
    manifest_path = output_dir / f"{model_id}_experiment_manifest_v1_2_1.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
