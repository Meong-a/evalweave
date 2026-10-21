from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common_v1_2 import format_choices, load_records


class FormalMultipleChoiceTask(TaskPlugin):
    """Multiple-choice task with A-Z labels and optional fixed few-shot context."""

    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["knowledge"]))
        samples: list[Sample] = []
        for index, row in enumerate(rows):
            choices = row["choices"]
            if not isinstance(choices, (dict, list)):
                raise ValueError(f"{name}:{index}: choices must be an object or list")
            labels = list(choices) if isinstance(choices, dict) else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
            answer = str(row["answer"]).strip().upper()
            if answer not in labels:
                raise ValueError(f"{name}:{index}: answer {answer!r} is not a choice")
            metadata = dict(row.get("metadata", {}))
            metadata.update({
                "source": str(row.get("source", name)),
                "row_index": index,
                "choice_labels": labels,
            })
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or f"{name}:{index:06d}"),
                    task_name=name,
                    capability=capabilities,
                    input={
                        "question": str(row["question"]),
                        "choices": format_choices(choices),
                        "few_shot": str(row.get("few_shot", "")),
                    },
                    reference={"answer": answer},
                    metadata=metadata,
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        template = str(
            self.config.get(
                "prompt_template",
                "{few_shot}\n\nQuestion: {question}\nOptions:\n{choices}\nAnswer with only the option letter:",
            )
        )
        return GenerationRequest(sample_id=sample.sample_id, prompt=template.format(**sample.input).strip())

    def parse_response(self, text: str, sample: Sample) -> str | None:
        labels = set(str(label).upper() for label in sample.metadata["choice_labels"])
        upper = text.upper().strip()
        patterns = [
            r"(?:FINAL\s+ANSWER|ANSWER|答案|选项)(?:\s+IS|是|为|\s*[:：])?\s*[（(]?\s*([A-Z])",
            r"[（(]\s*([A-Z])\s*[）)]",
            r"^\s*([A-Z])(?:\b|[.、，,：:])?",
        ]
        for pattern in patterns:
            match = re.search(pattern, upper, flags=re.IGNORECASE)
            if match and match.group(1).upper() in labels:
                return match.group(1).upper()
        return None

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        expected = sample.reference["answer"]
        return {
            "accuracy": float(prediction == expected),
            "valid_answer": float(prediction is not None),
        }


class SubmissionMultipleChoiceTask(FormalMultipleChoiceTask):
    """Unlabelled MCQ split: generates validated option labels without fake accuracy."""

    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["knowledge"]))
        samples: list[Sample] = []
        for index, row in enumerate(rows):
            choices = row["choices"]
            labels = list(choices) if isinstance(choices, dict) else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
            metadata = dict(row.get("metadata", {}))
            metadata.update({
                "source": str(row.get("source", name)),
                "row_index": index,
                "choice_labels": labels,
                "original_id": row.get("original_id", index),
            })
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or f"{name}:{index:06d}"),
                    task_name=name,
                    capability=capabilities,
                    input={
                        "question": str(row["question"]),
                        "choices": format_choices(choices),
                        "few_shot": str(row.get("few_shot", "")),
                    },
                    reference={"answer": None},
                    metadata=metadata,
                )
            )
        return samples

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        return {"generation_success": float(prediction is not None), "valid_answer": float(prediction is not None)}
