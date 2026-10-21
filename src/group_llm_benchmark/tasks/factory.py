from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .base import TaskPlugin
from .bbh import BBHTask
from .code_generation import CodeGenerationTask
from .formal_mcq import FormalMultipleChoiceTask, SubmissionMultipleChoiceTask
from .ifeval import IFEvalTask
from .mcq import MultipleChoiceTask
from .numeric_qa import NumericQATask
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
    if task_type in {"custom", "python_plugin"}:
        class_path = str(resolved.get("class_path", ""))
        if ":" not in class_path:
            raise ValueError("custom task requires class_path in 'module:ClassName' format")
        module_name, class_name = class_path.split(":", 1)
        module = importlib.import_module(module_name)
        task_class = getattr(module, class_name, None)
        if not isinstance(task_class, type) or not issubclass(task_class, TaskPlugin):
            raise ValueError(f"Custom task class must extend TaskPlugin: {class_path}")
        return task_class(resolved)
    tasks: dict[str, type[TaskPlugin]] = {
        "mcq": MultipleChoiceTask,
        "multiple_choice": MultipleChoiceTask,
        "mcq_formal": FormalMultipleChoiceTask,
        "mcq_submission": SubmissionMultipleChoiceTask,
        "bbh": BBHTask,
        "numeric_qa": NumericQATask,
        "ifeval": IFEvalTask,
        "code_generation": CodeGenerationTask,
        "text_generation": TextGenerationTask,
    }
    try:
        return tasks[task_type](resolved)
    except KeyError as exc:
        raise ValueError(f"Unknown task type: {task_type}") from exc
