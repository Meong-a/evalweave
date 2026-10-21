from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from group_llm_benchmark.reporting import render_run_html
from group_llm_benchmark.runners import summarize_predictions
from group_llm_benchmark.tasks.common import load_records
from group_llm_benchmark.utils.io import (
    environment_snapshot,
    hash_file,
    read_jsonl,
    safe_name,
    write_csv,
    write_json,
)


def import_legacy_mcq(
    dataset_path: str | Path,
    predictions_path: str | Path,
    output_root: str | Path,
    experiment_name: str,
    model_id: str,
    task_name: str,
    capabilities: list[str],
) -> Path:
    dataset_path = Path(dataset_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    dataset = load_records(dataset_path)
    legacy = read_jsonl(predictions_path)
    if len(dataset) != len(legacy):
        raise ValueError(f"Dataset and prediction lengths differ: {len(dataset)} != {len(legacy)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{safe_name(experiment_name)}_imported_v1"
    run_dir = Path(output_root).resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"Import target already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, (sample, prediction) in enumerate(zip(dataset, legacy)):
        answer = str(sample.get("answer", prediction.get("answer", ""))).strip().upper()
        parsed = prediction.get("prediction")
        parsed = str(parsed).strip().upper() if parsed is not None else None
        correct = bool(parsed == answer)
        source = str(sample.get("source", prediction.get("source", task_name)))
        rows.append(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "experiment_name": experiment_name,
                "model_id": model_id,
                "sample_id": f"{task_name}:{index:06d}",
                "task_name": task_name,
                "capability": capabilities,
                "input": {"question": sample.get("question"), "choices": sample.get("options", sample.get("choices"))},
                "reference": {"answer": answer},
                "metadata": {"source": source, "row_index": index, "imported_from": str(predictions_path)},
                "prompt": None,
                "raw_output": prediction.get("model_output", ""),
                "parsed_prediction": parsed,
                "scores": {"accuracy": float(correct), "valid_answer": float(parsed is not None)},
                "correct": correct,
                "latency_ms": 0.0,
                "error": None,
                "generation_config": {},
                "response_metadata": {"adapter": "legacy_import"},
            }
        )
    predictions_target = run_dir / "predictions_v1.jsonl"
    predictions_target.write_text("".join(__import__("json").dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    metrics = summarize_predictions(rows, run_id, "accuracy")
    config = {
        "schema_version": "1.0",
        "experiment_name": experiment_name,
        "model": {"type": "legacy_import", "id": model_id},
        "task": {"type": "mcq", "name": task_name, "dataset_path": str(dataset_path), "capabilities": capabilities},
        "metadata": {"legacy_predictions": str(predictions_path)},
    }
    write_json(run_dir / "resolved_config_v1.json", config)
    write_json(run_dir / "environment_v1.json", environment_snapshot())
    write_json(
        run_dir / "hashes_v1.json",
        {"dataset_sha256": hash_file(dataset_path), "legacy_predictions_sha256": hash_file(predictions_path)},
    )
    write_json(run_dir / "metrics_v1.json", metrics)
    write_csv(run_dir / "metrics_v1.csv", [{"scope": "overall", "name": "accuracy", "count": len(rows), "score": metrics["primary_score"]}])
    (run_dir / "report_v1.html").write_text(render_run_html(metrics, config), encoding="utf-8")
    return run_dir
