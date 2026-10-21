from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common_v1_2 import load_records


def normalize_target(value: str) -> str:
    value = " ".join(value.strip().split()).casefold()
    if re.fullmatch(r"\([a-z]\)", value):
        return value[1]
    if re.fullmatch(r"[a-z]", value):
        return value
    return value.rstrip(".。")


class BBHTask(TaskPlugin):
    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        samples = []
        for index, row in enumerate(rows):
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or f"{name}:{index:06d}"),
                    task_name=name,
                    capability=("complex_reasoning",),
                    input={"prompt": str(row["prompt"])},
                    reference={"target": normalize_target(str(row["reference"]))},
                    metadata={"source": str(row["source"]), "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        return GenerationRequest(sample_id=sample.sample_id, prompt=sample.input["prompt"])

    def parse_response(self, text: str, sample: Sample) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        matches = re.findall(r"(?:so\s+)?the\s+answer\s+is\s*[:：]?\s*(.+)", stripped, flags=re.I)
        candidate = matches[-1] if matches else stripped.splitlines()[-1]
        candidate = re.sub(r"^(?:final\s+answer|answer)\s*[:：]\s*", "", candidate, flags=re.I)
        return normalize_target(candidate)

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        expected = sample.reference["target"]
        return {
            "exact_match": float(prediction == expected),
            "valid_answer": float(prediction is not None),
        }
