from __future__ import annotations

from pathlib import Path
from typing import Any

from group_llm_benchmark.schemas import ExperimentConfig
from group_llm_benchmark.utils.io import read_config


def _resolve_component(value: Any, base_dir: Path, name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        component = read_config(path)
        component.pop("_config_path", None)
        return component
    raise ValueError(f"{name} must be an object or a config path string")


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = read_config(path)
    config_path = Path(str(raw["_config_path"]))
    base_dir = config_path.parent
    if "model" not in raw and "model_ref" in raw:
        raw["model"] = raw.pop("model_ref")
    if "task" not in raw and "task_ref" in raw:
        raw["task"] = raw.pop("task_ref")
    raw["model"] = _resolve_component(raw.get("model"), base_dir, "model")
    raw["task"] = _resolve_component(raw.get("task"), base_dir, "task")
    return ExperimentConfig.from_dict(raw)
