from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TaskPlugin
from .mcq import MultipleChoiceTask
from .text_generation import TextGenerationTask


def create_task(config: dict[str, Any], base_dir: Path | None = None) -> TaskPlugin:
    resolved = dict(config)
    if "dataset_path" not in resolved:
        raise ValueError("task requires dataset_path")
    dataset_path = Path(str(resolved["dataset_path"]))
    if not dataset_path.is_absolute():
        dataset_path = ((base_dir or Path.cwd()) / dataset_path).resolve()
    resolved["dataset_path"] = str(dataset_path)
    task_type = str(resolved.get("type", "")).lower()
    tasks: dict[str, type[TaskPlugin]] = {
        "mcq": MultipleChoiceTask,
        "multiple_choice": MultipleChoiceTask,
        "text_generation": TextGenerationTask,
    }
    try:
        return tasks[task_type](resolved)
    except KeyError as exc:
        raise ValueError(f"Unknown task type: {task_type}") from exc
