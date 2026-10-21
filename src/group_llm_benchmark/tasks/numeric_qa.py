from __future__ import annotations

import decimal
import re
from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common_v1_2 import load_records


_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def canonical_number(value: str) -> str | None:
    value = value.replace(",", "").strip().rstrip(".。")
    try:
        number = decimal.Decimal(value)
    except decimal.InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


class NumericQATask(TaskPlugin):
    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        capabilities = tuple(str(item) for item in self.config.get("capabilities", ["mathematical_reasoning"]))
        samples = []
        for index, row in enumerate(rows):
            reference = canonical_number(str(row["reference"]))
            if reference is None:
                raise ValueError(f"{name}:{index}: invalid numeric reference")
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or f"{name}:{index:06d}"),
                    task_name=name,
                    capability=capabilities,
                    input={"text": str(row["input"]), "few_shot": str(row.get("few_shot", ""))},
                    reference={"number": reference},
                    metadata={"source": str(row.get("source", name)), "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        template = str(self.config.get("prompt_template", "{few_shot}\n\n{text}\nFinal numeric answer:"))
        return GenerationRequest(sample_id=sample.sample_id, prompt=template.format(**sample.input).strip())

    def parse_response(self, text: str, sample: Sample) -> str | None:
        candidates = _NUMBER.findall(text)
        return canonical_number(candidates[-1]) if candidates else None

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        expected = sample.reference["number"]
        return {
            "numeric_exact_match": float(prediction == expected),
            "valid_answer": float(prediction is not None),
        }
