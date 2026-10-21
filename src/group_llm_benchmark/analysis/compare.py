from __future__ import annotations

import math
import random
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from group_llm_benchmark.reporting import render_comparison_html
from group_llm_benchmark.utils.io import read_jsonl, safe_name, utc_now, write_csv, write_json


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_statistics(
    baseline: list[float], candidate: list[float], seed: int = 42, bootstrap_samples: int = 2000
) -> dict[str, Any]:
    deltas = [right - left for left, right in zip(baseline, candidate)]
    rng = random.Random(seed)
    estimates: list[float] = []
    if deltas:
        for _ in range(bootstrap_samples):
            estimates.append(_mean([deltas[rng.randrange(len(deltas))] for _ in deltas]))
        estimates.sort()
        low = estimates[int(0.025 * (len(estimates) - 1))]
        high = estimates[int(0.975 * (len(estimates) - 1))]
    else:
        low = high = 0.0
    baseline_only = sum(left >= 1.0 and right < 1.0 for left, right in zip(baseline, candidate))
    candidate_only = sum(left < 1.0 and right >= 1.0 for left, right in zip(baseline, candidate))
    discordant = baseline_only + candidate_only
    statistic = ((abs(baseline_only - candidate_only) - 1) ** 2 / discordant) if discordant else 0.0
    return {
        "paired_bootstrap_95_ci": [low, high],
        "bootstrap_samples": bootstrap_samples,
        "mcnemar": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "discordant": discordant,
            "continuity_corrected_chi_square": statistic,
            "approximate_p_value": math.erfc(math.sqrt(statistic / 2)) if discordant else 1.0,
        },
    }


def compare_runs(
    baseline_run: str | Path,
    candidate_run: str | Path,
    output_root: str | Path,
    seed: int = 42,
    bootstrap_samples: int = 2000,
) -> Path:
    baseline_run = Path(baseline_run).resolve()
    candidate_run = Path(candidate_run).resolve()
    baseline_rows = read_jsonl(baseline_run / "predictions_v1.jsonl")
    candidate_rows = read_jsonl(candidate_run / "predictions_v1.jsonl")
    baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_rows}
    if set(baseline_by_id) != set(candidate_by_id):
        missing_candidate = sorted(set(baseline_by_id) - set(candidate_by_id))[:10]
        missing_baseline = sorted(set(candidate_by_id) - set(baseline_by_id))[:10]
        raise ValueError(
            f"Sample sets differ; missing in candidate={missing_candidate}, missing in baseline={missing_baseline}"
        )
    baseline_metrics = json.loads((baseline_run / "metrics_v1.json").read_text(encoding="utf-8"))
    candidate_metrics = json.loads((candidate_run / "metrics_v1.json").read_text(encoding="utf-8"))
    baseline_hashes = json.loads((baseline_run / "hashes_v1.json").read_text(encoding="utf-8"))
    candidate_hashes = json.loads((candidate_run / "hashes_v1.json").read_text(encoding="utf-8"))
    primary = str(baseline_metrics["primary_metric"])
    if candidate_metrics.get("primary_metric") != primary:
        raise ValueError("Primary metrics differ; comparison would be invalid")
    ids = sorted(baseline_by_id)
    baseline_values = [float(baseline_by_id[item].get("scores", {}).get(primary, 0.0)) for item in ids]
    candidate_values = [float(candidate_by_id[item].get("scores", {}).get(primary, 0.0)) for item in ids]
    transitions = {"both_correct": 0, "improved": 0, "regressed": 0, "both_incorrect": 0}
    changed_rows: list[dict[str, Any]] = []
    capability_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"baseline": [], "candidate": []})
    for sample_id, left, right in zip(ids, baseline_values, candidate_values):
        if left >= 1.0 and right >= 1.0:
            transition = "both_correct"
        elif left < 1.0 and right >= 1.0:
            transition = "improved"
        elif left >= 1.0 and right < 1.0:
            transition = "regressed"
        else:
            transition = "both_incorrect"
        transitions[transition] += 1
        baseline_row = baseline_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        for capability in sorted(set(baseline_row.get("capability", [])) | set(candidate_row.get("capability", []))):
            capability_values[str(capability)]["baseline"].append(left)
            capability_values[str(capability)]["candidate"].append(right)
        if transition in {"improved", "regressed"}:
            changed_rows.append(
                {
                    "sample_id": sample_id,
                    "transition": transition,
                    "reference": baseline_row.get("reference"),
                    "baseline_prediction": baseline_row.get("parsed_prediction"),
                    "candidate_prediction": candidate_row.get("parsed_prediction"),
                    "source": baseline_row.get("metadata", {}).get("source"),
                }
            )
    capability_matrix = {}
    for capability, values in sorted(capability_values.items()):
        left = _mean(values["baseline"])
        right = _mean(values["candidate"])
        capability_matrix[capability] = {
            "count": len(values["baseline"]),
            "baseline": left,
            "candidate": right,
            "delta": right - left,
            "relative_change": ((right - left) / left) if left else None,
        }
    baseline_score = _mean(baseline_values)
    candidate_score = _mean(candidate_values)
    result = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "primary_metric": primary,
        "compatibility": {
            "same_sample_ids": True,
            "same_primary_metric": True,
            "same_dataset_hash": baseline_hashes.get("dataset_sha256") == candidate_hashes.get("dataset_sha256"),
            "same_prompt_hash": baseline_hashes.get("prompt_sha256") == candidate_hashes.get("prompt_sha256"),
            "warning": "Dataset or prompt hashes differ; interpret scores cautiously."
            if baseline_hashes.get("dataset_sha256") != candidate_hashes.get("dataset_sha256")
            or baseline_hashes.get("prompt_sha256") != candidate_hashes.get("prompt_sha256")
            else None,
        },
        "overall": {
            "count": len(ids),
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
            "relative_change": ((candidate_score - baseline_score) / baseline_score) if baseline_score else None,
        },
        "transitions": transitions,
        "capability_matrix": capability_matrix,
        "statistics": _paired_statistics(baseline_values, candidate_values, seed, bootstrap_samples),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = safe_name(f"{candidate_run.name}_vs_{baseline_run.name}")
    output_dir = Path(output_root).resolve() / f"{timestamp}_{name}"
    if output_dir.exists():
        raise FileExistsError(f"Comparison directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "comparison_v1.json", result)
    write_csv(
        output_dir / "capability_matrix_v1.csv",
        [{"capability": capability, **values} for capability, values in capability_matrix.items()],
    )
    write_json(output_dir / "changed_samples_v1.json", changed_rows)
    write_csv(output_dir / "changed_samples_v1.csv", changed_rows)
    (output_dir / "comparison_report_v1.html").write_text(render_comparison_html(result), encoding="utf-8")
    return output_dir
