from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.analysis import build_leaderboard, build_suite_matrix
from group_llm_benchmark.config import load_experiment
from group_llm_benchmark.runners import BenchmarkRunner
from group_llm_benchmark.utils.io import read_config


def run_suite(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_config(path)
    if config.get("schema_version") != "1.0":
        raise ValueError("Unsupported suite schema_version")
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("suite experiments must be a non-empty list")
    run_dirs: list[Path] = []
    for experiment in experiments:
        experiment_path = Path(str(experiment))
        if not experiment_path.is_absolute():
            experiment_path = (path.parent / experiment_path).resolve()
        run_dirs.append(BenchmarkRunner(load_experiment(experiment_path)).run())
    mode = str(config.get("mode", "leaderboard"))
    output_root_value = config.get("output_root", "../../leaderboards" if mode == "leaderboard" else "../../suite_reports")
    output_root = Path(str(output_root_value))
    if not output_root.is_absolute():
        output_root = (path.parent / output_root).resolve()
    if mode == "leaderboard":
        report = build_leaderboard(run_dirs, output_root, str(config.get("suite_name", path.stem)))
    elif mode == "matrix":
        report = build_suite_matrix(run_dirs, output_root, str(config.get("suite_name", path.stem)))
    else:
        raise ValueError("suite mode must be 'leaderboard' or 'matrix'")
    return {
        "suite_name": config.get("suite_name"),
        "mode": mode,
        "runs": [str(item) for item in run_dirs],
        "report": str(report),
    }
