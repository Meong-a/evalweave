from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample
from group_llm_benchmark.tasks.base import TaskPlugin
from group_llm_benchmark.tasks.common_v1_2 import load_records


class KeywordCoverageTask(TaskPlugin):
    """Example custom task: score how many required keywords appear in the answer."""

    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["domain_generation"]))
        samples = []
        for index, row in enumerate(rows):
            keywords = [str(item).strip() for item in row["reference_keywords"] if str(item).strip()]
            if not keywords:
                raise ValueError(f"{name}:{index}: reference_keywords cannot be empty")
            samples.append(Sample(
                sample_id=str(row.get("id") or f"{name}:{index:06d}"),
                task_name=name,
                capability=capabilities,
                input={"text": str(row["input"])},
                reference={"keywords": keywords},
                metadata={"source": str(row.get("source", name)), "row_index": index},
            ))
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        template = str(self.config.get("prompt_template", "{text}"))
        return GenerationRequest(sample_id=sample.sample_id, prompt=template.format(**sample.input))

    def parse_response(self, text: str, sample: Sample) -> str:
        return text.strip()

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        value = str(prediction).casefold()
        keywords = [str(item).casefold() for item in sample.reference["keywords"]]
        matched = sum(keyword in value for keyword in keywords)
        recall = matched / len(keywords)
        return {"keyword_recall": recall, "all_keywords": float(matched == len(keywords))}
