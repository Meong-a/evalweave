from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample

from .base import TaskPlugin
from .common_v1_2 import load_records


class IFEvalTask(TaskPlugin):
    def load_samples(self) -> list[Sample]:
        path = Path(str(self.config["dataset_path"]))
        rows = load_records(path)
        name = str(self.config.get("name", path.stem))
        samples = []
        for index, row in enumerate(rows):
            samples.append(
                Sample(
                    sample_id=str(row.get("id") or f"ifeval:{row.get('key', index)}"),
                    task_name=name,
                    capability=("instruction_following",),
                    input={"prompt": str(row["prompt"])},
                    reference={
                        "key": int(row["key"]),
                        "instruction_id_list": list(row["instruction_id_list"]),
                        "kwargs": list(row["kwargs"]),
                    },
                    metadata={"source": "ifeval/official", "row_index": index},
                )
            )
        return samples

    def build_request(self, sample: Sample) -> GenerationRequest:
        return GenerationRequest(sample_id=sample.sample_id, prompt=sample.input["prompt"])

    def parse_response(self, text: str, sample: Sample) -> str:
        return text

    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        try:
            from instruction_following_eval import evaluation_lib
        except ImportError as exc:
            raise RuntimeError(
                "IFEval official scorer is unavailable. Run scripts/install_ifeval_deps_v1_2.sh "
                "and ensure src/instruction_following_eval exists."
            ) from exc
        inp = evaluation_lib.InputExample(
            key=sample.reference["key"],
            instruction_id_list=sample.reference["instruction_id_list"],
            prompt=sample.input["prompt"],
            kwargs=sample.reference["kwargs"],
        )
        mapping = {sample.input["prompt"]: str(prediction)}
        strict = evaluation_lib.test_instruction_following_strict(inp, mapping)
        loose = evaluation_lib.test_instruction_following_loose(inp, mapping)
        return {
            "strict_prompt_accuracy": float(strict.follow_all_instructions),
            "loose_prompt_accuracy": float(loose.follow_all_instructions),
            "strict_instruction_accuracy": (
                sum(strict.follow_instruction_list) / len(strict.follow_instruction_list)
            ),
            "loose_instruction_accuracy": (
                sum(loose.follow_instruction_list) / len(loose.follow_instruction_list)
            ),
        }
