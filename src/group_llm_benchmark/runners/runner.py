from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from group_llm_benchmark.adapters import create_adapter
from group_llm_benchmark.reporting import render_run_html
from group_llm_benchmark.schemas import ExperimentConfig, GenerationResponse
from group_llm_benchmark.tasks import create_task
from group_llm_benchmark.utils.io import (
    append_jsonl,
    environment_snapshot,
    hash_file,
    hash_value,
    read_jsonl,
    safe_name,
    utc_now,
    write_csv,
    write_json,
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_predictions(rows: list[dict[str, Any]], run_id: str, primary_metric: str) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    capability_scores: dict[str, list[float]] = defaultdict(list)
    task_scores: dict[str, list[float]] = defaultdict(list)
    source_scores: dict[str, list[float]] = defaultdict(list)
    error_count = 0
    invalid_count = 0
    for row in rows:
        error_count += int(bool(row.get("error")))
        invalid_count += int(row.get("parsed_prediction") is None)
        for name, value in row.get("scores", {}).items():
            scores[name].append(float(value))
        primary = float(row.get("scores", {}).get(primary_metric, 0.0))
        task_scores[str(row.get("task_name", "unknown"))].append(primary)
        source_scores[str(row.get("metadata", {}).get("source", "unknown"))].append(primary)
        for capability in row.get("capability", []):
            capability_scores[str(capability)].append(primary)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": utc_now(),
        "count": len(rows),
        "error_count": error_count,
        "invalid_count": invalid_count,
        "primary_metric": primary_metric,
        "primary_score": _mean(scores.get(primary_metric, [])),
        "scores": {name: _mean(values) for name, values in sorted(scores.items())},
        "per_task": {name: {"count": len(values), "score": _mean(values)} for name, values in sorted(task_scores.items())},
        "per_capability": {
            name: {"count": len(values), "score": _mean(values)} for name, values in sorted(capability_scores.items())
        },
        "per_source": {
            name: {"count": len(values), "score": _mean(values)} for name, values in sorted(source_scores.items())
        },
        "total_latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in rows),
    }


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig, run_dir: str | Path | None = None) -> None:
        self.config = config
        self.config_dict = config.to_dict()
        self.config_hash = hash_value(self.config_dict)
        if run_dir:
            self.run_dir = Path(run_dir).resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = safe_name(config.experiment_name)
            self.run_dir = config.output_root / f"{timestamp}_{name}_{self.config_hash[:8]}"

    def _prepare(self, dataset_path: Path, prompt_template: str) -> None:
        if self.run_dir.exists() and not self.config.resume:
            raise FileExistsError(f"Run directory already exists and resume=false: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        hashes_path = self.run_dir / "hashes_v1.json"
        hashes = {
            "config_sha256": self.config_hash,
            "dataset_sha256": hash_file(dataset_path),
            "prompt_sha256": hash_value(prompt_template),
        }
        if hashes_path.exists():
            previous = json.loads(hashes_path.read_text(encoding="utf-8"))
            if previous != hashes:
                raise ValueError("Cannot resume: config, dataset, or prompt hash changed")
        write_json(self.run_dir / "resolved_config_v1.json", self.config_dict)
        write_json(self.run_dir / "environment_v1.json", environment_snapshot())
        write_json(hashes_path, hashes)

    def run(self) -> Path:
        base_dir = self.config.config_path.parent if self.config.config_path else Path.cwd()
        task = create_task(self.config.task, base_dir=base_dir)
        samples = task.load_samples()
        dataset_path = Path(str(task.config["dataset_path"]))
        prompt_template = str(task.config.get("prompt_template", ""))
        self._prepare(dataset_path, prompt_template)
        predictions_path = self.run_dir / "predictions_v1.jsonl"
        existing = read_jsonl(predictions_path) if predictions_path.exists() else []
        completed = {str(row["sample_id"]) for row in existing}
        pending = [sample for sample in samples if sample.sample_id not in completed]
        log_path = self.run_dir / "run_log_v1.jsonl"
        append_jsonl(log_path, [{"event": "run_started", "at": utc_now(), "pending": len(pending)}])
        adapter = create_adapter(self.config.model)
        with adapter:
            for start in range(0, len(pending), self.config.batch_size):
                batch = pending[start : start + self.config.batch_size]
                requests = [task.build_request(sample) for sample in batch]
                try:
                    responses = adapter.generate(requests, self.config.generation)
                    response_by_id = {response.sample_id: response for response in responses}
                except Exception as exc:
                    response_by_id = {
                        request.sample_id: GenerationResponse(
                            sample_id=request.sample_id,
                            text="",
                            latency_ms=0.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        for request in requests
                    }
                output_rows: list[dict[str, Any]] = []
                for sample, request in zip(batch, requests):
                    response = response_by_id.get(sample.sample_id)
                    if response is None:
                        response = GenerationResponse(sample.sample_id, "", 0.0, error="adapter returned no response")
                    prediction = None if response.error else task.parse_response(response.text, sample)
                    scores = task.score(prediction, sample) if not response.error else {}
                    primary_metric = str(task.config.get("primary_metric", "accuracy"))
                    output_rows.append(
                        {
                            "schema_version": "1.0",
                            "run_id": self.run_dir.name,
                            "experiment_name": self.config.experiment_name,
                            "model_id": str(self.config.model.get("id", self.config.model.get("model", "model"))),
                            "sample_id": sample.sample_id,
                            "task_name": sample.task_name,
                            "capability": list(sample.capability),
                            "input": sample.input,
                            "reference": sample.reference,
                            "metadata": sample.metadata,
                            "prompt": request.prompt,
                            "raw_output": response.text,
                            "parsed_prediction": prediction,
                            "scores": scores,
                            "correct": bool(scores.get(primary_metric, 0.0) >= 1.0),
                            "latency_ms": response.latency_ms,
                            "error": response.error,
                            "generation_config": self.config.generation,
                            "response_metadata": response.metadata,
                        }
                    )
                append_jsonl(predictions_path, output_rows)
                append_jsonl(
                    log_path,
                    [{"event": "batch_completed", "at": utc_now(), "completed": min(start + len(batch), len(pending))}],
                )
        rows = read_jsonl(predictions_path)
        primary_metric = str(task.config.get("primary_metric", "accuracy"))
        metrics = summarize_predictions(rows, self.run_dir.name, primary_metric)
        write_json(self.run_dir / "metrics_v1.json", metrics)
        write_csv(
            self.run_dir / "metrics_v1.csv",
            [{"scope": "overall", "name": primary_metric, "count": metrics["count"], "score": metrics["primary_score"]}]
            + [
                {"scope": "capability", "name": name, **values}
                for name, values in metrics["per_capability"].items()
            ],
        )
        (self.run_dir / "report_v1.html").write_text(
            render_run_html(metrics, self.config_dict), encoding="utf-8"
        )
        append_jsonl(log_path, [{"event": "run_completed", "at": utc_now(), "count": len(rows)}])
        return self.run_dir
