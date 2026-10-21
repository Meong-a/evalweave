from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require(mapping: dict[str, Any], name: str, expected_type: type) -> Any:
    value = mapping.get(name)
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be {expected_type.__name__}")
    return value


@dataclass(frozen=True)
class Sample:
    sample_id: str
    task_name: str
    capability: tuple[str, ...]
    input: dict[str, Any]
    reference: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.capability:
            raise ValueError(f"sample {self.sample_id} must declare at least one capability")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "capability": list(self.capability),
            "input": self.input,
            "reference": self.reference,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GenerationRequest:
    sample_id: str
    prompt: str
    messages: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class GenerationResponse:
    sample_id: str
    text: str
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    experiment_name: str
    model: dict[str, Any]
    task: dict[str, Any]
    generation: dict[str, Any]
    output_root: Path
    seed: int = 42
    batch_size: int = 1
    resume: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        schema_version = _require(value, "schema_version", str)
        if schema_version != "1.0":
            raise ValueError(f"Unsupported schema_version: {schema_version}")
        config_path = Path(value["_config_path"]).resolve() if value.get("_config_path") else None
        base = config_path.parent if config_path else Path.cwd()
        output_value = value.get("output_root", "runs")
        output_root = Path(output_value)
        if not output_root.is_absolute():
            output_root = (base / output_root).resolve()
        batch_size = int(value.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return cls(
            schema_version=schema_version,
            experiment_name=_require(value, "experiment_name", str),
            model=_require(value, "model", dict),
            task=_require(value, "task", dict),
            generation=dict(value.get("generation", {})),
            output_root=output_root,
            seed=int(value.get("seed", 42)),
            batch_size=batch_size,
            resume=bool(value.get("resume", True)),
            metadata=dict(value.get("metadata", {})),
            config_path=config_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "model": self.model,
            "task": self.task,
            "generation": self.generation,
            "output_root": str(self.output_root),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "resume": self.resume,
            "metadata": self.metadata,
            "config_path": str(self.config_path) if self.config_path else None,
        }
