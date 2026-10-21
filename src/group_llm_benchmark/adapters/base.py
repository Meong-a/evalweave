from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from group_llm_benchmark.schemas import GenerationRequest, GenerationResponse


class ModelAdapter(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def load(self) -> None:
        return None

    @abstractmethod
    def generate(
        self, requests: Sequence[GenerationRequest], generation_config: dict[str, Any]
    ) -> list[GenerationResponse]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "ModelAdapter":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
