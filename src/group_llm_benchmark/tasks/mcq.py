from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common import format_choices, load_records


class MultipleChoiceTask(TaskPlugin):
    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["knowledge"]))
        question_field = str(self.config.get("question_field", "question"))
        choices_field = str(self.config.get("choices_field", "choices"))
        answer_field = str(self.config.get("answer_field", "answer"))
        source_field = str(self.config.get("source_field", "source"))
        id_field = self.config.get("id_field")
        limit = self.config.get("limit")
        if limit is not None:
            rows = rows[: int(limit)]
        samples: list[Sample] = []
        for index, row in enumerate(rows):
            sample_id = str(row.get(str(id_field), "")) if id_field else ""
            if not sample_id:
                sample_id = f"{name}:{index:06d}"
            source = str(row.get(source_field, name))
            samples.append(
                Sample(
                    sample_id=sample_id,
                    task_name=name,
                    capability=capabilities,
                    input={
                        "question": str(row[question_field]),
                        "choices": format_choices(row[choices_field]),
                    },
                    reference={"answer": str(row[answer_field]).strip().upper()},
                    metadata={"source": source, "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        template = str(
            self.config.get(
                "prompt_template",
                "Select the single correct answer. Reply with only the option letter.\n"
                "Question: {question}\nOptions:\n{choices}\nAnswer:",
            )
        )
        prompt = template.format(**sample.input)
        return GenerationRequest(sample_id=sample.sample_id, prompt=prompt)

    def parse_response(self, text: str, sample: Sample) -> str | None:
        patterns = self.config.get(
            "answer_patterns",
            [
                r"(?:answer|答案|选项)(?:\s+is|是|为|：|:)*\s*[（(]?\s*([A-F])",
                r"[（(]\s*([A-F])\s*[）)]",
                r"^\s*([A-F])(?:\b|[.、，,：:])?",
            ],
        )
        upper = text.upper()
        for pattern in patterns:
            match = re.search(str(pattern), upper, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        expected = sample.reference["answer"]
        return {
            "accuracy": float(prediction == expected),
            "valid_answer": float(prediction is not None),
        }
