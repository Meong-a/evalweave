from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from group_llm_benchmark.reporting import render_leaderboard_html, render_suite_matrix_html
from group_llm_benchmark.utils.io import safe_name, utc_now, write_csv, write_json


def build_leaderboard(run_dirs: list[str | Path], output_root: str | Path, name: str = "leaderboard_v1") -> Path:
    if len(run_dirs) < 2:
        raise ValueError("A leaderboard requires at least two runs")
    ranking: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    primary_metric: str | None = None
    reference_dataset_hash: str | None = None
    reference_prompt_hash: str | None = None
    for run_value in run_dirs:
        run_dir = Path(run_value).resolve()
        metrics = json.loads((run_dir / "metrics_v1.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config_v1.json").read_text(encoding="utf-8"))
        hashes = json.loads((run_dir / "hashes_v1.json").read_text(encoding="utf-8"))
        current_metric = str(metrics["primary_metric"])
        if primary_metric is None:
            primary_metric = current_metric
        elif primary_metric != current_metric:
            raise ValueError("All leaderboard runs must use the same primary metric")
        if reference_dataset_hash is None:
            reference_dataset_hash = hashes.get("dataset_sha256")
            reference_prompt_hash = hashes.get("prompt_sha256")
        elif reference_dataset_hash != hashes.get("dataset_sha256") or reference_prompt_hash != hashes.get("prompt_sha256"):
            raise ValueError("Leaderboard runs must use the same dataset and prompt hashes")
        model_id = str(config.get("model", {}).get("id", config.get("experiment_name", run_dir.name)))
        ranking.append(
            {
                "run_id": run_dir.name,
                "model_id": model_id,
                "experiment": config.get("experiment_name"),
                "count": metrics.get("count"),
                "primary_metric": current_metric,
                "score": metrics.get("primary_score"),
                "errors": metrics.get("error_count"),
                "invalid": metrics.get("invalid_count"),
            }
        )
        for capability, values in metrics.get("per_capability", {}).items():
            capability_rows.append(
                {
                    "model_id": model_id,
                    "run_id": run_dir.name,
                    "capability": capability,
                    "count": values.get("count"),
                    "score": values.get("score"),
                }
            )
    ranking.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root).resolve() / f"{timestamp}_{safe_name(name)}"
    if output_dir.exists():
        raise FileExistsError(f"Leaderboard directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    result = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "name": name,
        "primary_metric": primary_metric,
        "ranking": ranking,
        "capability_matrix": capability_rows,
    }
    write_json(output_dir / "leaderboard_v1.json", result)
    write_csv(output_dir / "leaderboard_v1.csv", ranking)
    write_csv(output_dir / "capability_matrix_v1.csv", capability_rows)
    (output_dir / "leaderboard_report_v1.html").write_text(render_leaderboard_html(result), encoding="utf-8")
    return output_dir


def build_suite_matrix(run_dirs: list[str | Path], output_root: str | Path, name: str = "suite_matrix_v1") -> Path:
    if not run_dirs:
        raise ValueError("A suite matrix requires at least one run")
    run_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    for run_value in run_dirs:
        run_dir = Path(run_value).resolve()
        metrics = json.loads((run_dir / "metrics_v1.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config_v1.json").read_text(encoding="utf-8"))
        model_id = str(config.get("model", {}).get("id", config.get("experiment_name", run_dir.name)))
        task_name = str(config.get("task", {}).get("name", next(iter(metrics.get("per_task", {})), "unknown")))
        run_rows.append(
            {
                "run_id": run_dir.name,
                "model_id": model_id,
                "task": task_name,
                "primary_metric": metrics.get("primary_metric"),
                "score": metrics.get("primary_score"),
                "count": metrics.get("count"),
                "errors": metrics.get("error_count"),
                "invalid": metrics.get("invalid_count"),
            }
        )
        for capability, values in metrics.get("per_capability", {}).items():
            capability_rows.append(
                {
                    "model_id": model_id,
                    "task": task_name,
                    "capability": capability,
                    "primary_metric": metrics.get("primary_metric"),
                    "count": values.get("count"),
                    "score": values.get("score"),
                    "run_id": run_dir.name,
                }
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root).resolve() / f"{timestamp}_{safe_name(name)}"
    if output_dir.exists():
        raise FileExistsError(f"Suite matrix directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    result = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "name": name,
        "runs": run_rows,
        "capability_matrix": capability_rows,
    }
    write_json(output_dir / "suite_matrix_v1.json", result)
    write_csv(output_dir / "suite_runs_v1.csv", run_rows)
    write_csv(output_dir / "capability_matrix_v1.csv", capability_rows)
    (output_dir / "suite_matrix_report_v1.html").write_text(render_suite_matrix_html(result), encoding="utf-8")
    return output_dir
