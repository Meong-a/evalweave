from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from group_llm_benchmark.analysis import build_leaderboard, compare_runs
from group_llm_benchmark.config import load_experiment
from group_llm_benchmark.importers import import_legacy_mcq
from group_llm_benchmark.runners import BenchmarkRunner
from group_llm_benchmark.suites import run_suite
from group_llm_benchmark.tasks import create_task
from group_llm_benchmark.utils.io import read_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbench", description="Group-wide reusable LLM benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate and resolve an experiment config")
    validate.add_argument("--config", required=True, type=Path)

    validate_task = subparsers.add_parser("validate-task", help="Load every row in a task config without running a model")
    validate_task.add_argument("--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="Run one model on one task")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--run-dir", type=Path)

    suite = subparsers.add_parser("run-suite", help="Run multiple experiments and build a leaderboard")
    suite.add_argument("--config", required=True, type=Path)

    compare = subparsers.add_parser("compare", help="Compare aligned benchmark runs")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output-root", required=True, type=Path)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)

    leaderboard = subparsers.add_parser("leaderboard", help="Build a multi-model leaderboard from completed runs")
    leaderboard.add_argument("--run", action="append", required=True, type=Path)
    leaderboard.add_argument("--output-root", required=True, type=Path)
    leaderboard.add_argument("--name", default="leaderboard_v1")

    legacy = subparsers.add_parser("import-legacy-mcq", help="Import aligned legacy MCQ predictions")
    legacy.add_argument("--dataset", required=True, type=Path)
    legacy.add_argument("--predictions", required=True, type=Path)
    legacy.add_argument("--output-root", required=True, type=Path)
    legacy.add_argument("--experiment-name", required=True)
    legacy.add_argument("--model-id", required=True)
    legacy.add_argument("--task-name", required=True)
    legacy.add_argument("--capability", action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_experiment(args.config)
            print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        elif args.command == "validate-task":
            task_path = args.config.resolve()
            task_config = read_config(task_path)
            task_config.pop("_config_path", None)
            task = create_task(task_config, base_dir=task_path.parent)
            samples = task.load_samples()
            if not samples:
                raise ValueError("task contains no samples")
            first_request = task.build_request(samples[0])
            print(json.dumps({
                "status": "PASS",
                "task_name": samples[0].task_name,
                "sample_count": len(samples),
                "first_sample_id": samples[0].sample_id,
                "first_prompt_preview": first_request.prompt[:200],
            }, ensure_ascii=False, indent=2))
        elif args.command == "run":
            config = load_experiment(args.config)
            result = BenchmarkRunner(config, run_dir=args.run_dir).run()
            print(result)
        elif args.command == "run-suite":
            print(json.dumps(run_suite(args.config), ensure_ascii=False, indent=2))
        elif args.command == "compare":
            result = compare_runs(
                args.baseline,
                args.candidate,
                args.output_root,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            print(result)
        elif args.command == "leaderboard":
            result = build_leaderboard(args.run, args.output_root, args.name)
            print(result)
        elif args.command == "import-legacy-mcq":
            result = import_legacy_mcq(
                args.dataset,
                args.predictions,
                args.output_root,
                args.experiment_name,
                args.model_id,
                args.task_name,
                args.capability,
            )
            print(result)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
