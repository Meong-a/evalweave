from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from group_llm_benchmark.schemas import GenerationRequest, Sample


class TaskPlugin(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def load_samples(self) -> list[Sample]:
        raise NotImplementedError

    @abstractmethod
    def build_request(self, sample: Sample) -> GenerationRequest:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, text: str, sample: Sample) -> Any:
        raise NotImplementedError

    @abstractmethod
    def score(self, prediction: Any, sample: Sample) -> dict[str, float]:
        raise NotImplementedError
