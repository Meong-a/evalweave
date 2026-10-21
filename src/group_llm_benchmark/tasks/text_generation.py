from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample
from group_llm_benchmark.scorers import score_text

from .base import TaskPlugin
from .common import load_records


class TextGenerationTask(TaskPlugin):
    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["generation"]))
        input_field = str(self.config.get("input_field", "input"))
        reference_field = str(self.config.get("reference_field", "reference"))
        id_field = self.config.get("id_field")
        source_field = str(self.config.get("source_field", "source"))
        limit = self.config.get("limit")
        if limit is not None:
            rows = rows[: int(limit)]
        samples: list[Sample] = []
        for index, row in enumerate(rows):
            sample_id = str(row.get(str(id_field), "")) if id_field else ""
            if not sample_id:
                sample_id = f"{name}:{index:06d}"
            samples.append(
                Sample(
                    sample_id=sample_id,
                    task_name=name,
                    capability=capabilities,
                    input={"text": str(row[input_field])},
                    reference={"text": str(row[reference_field])},
                    metadata={"source": str(row.get(source_field, name)), "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        template = str(self.config.get("prompt_template", "{text}"))
        return GenerationRequest(sample_id=sample.sample_id, prompt=template.format(**sample.input))

    def parse_response(self, text: str, sample: Sample) -> str:
        return text.strip()

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        metrics = [str(item) for item in self.config.get("metrics", ["exact_match"])]
        return score_text(str(prediction), str(sample.reference["text"]), metrics)
