from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common_v1_2 import load_records


class CodeGenerationTask(TaskPlugin):
    """Generation-only task. Functional correctness is scored in a sandbox."""

    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        samples = []
        for index, row in enumerate(rows):
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or row["task_id"]),
                    task_name=name,
                    capability=("code_generation",),
                    input={"prompt": str(row["prompt"])},
                    reference={"task_id": str(row["task_id"])},
                    metadata={"source": "humaneval/official", "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        return GenerationRequest(sample_id=sample.sample_id, prompt=sample.input["prompt"])

    def parse_response(self, text: str, sample: Sample) -> str | None:
        return text if text.strip() else None

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        return {"generation_success": float(prediction is not None)}
