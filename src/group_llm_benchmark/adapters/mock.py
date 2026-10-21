from __future__ import annotations

import re
import time
from typing import Any, Sequence

from group_llm_benchmark.schemas import GenerationRequest, GenerationResponse

from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    """Deterministic adapter for CI, smoke tests, and examples."""

    def generate(
        self, requests: Sequence[GenerationRequest], generation_config: dict[str, Any]
    ) -> list[GenerationResponse]:
        answers = dict(self.config.get("answers", {}))
        default = str(self.config.get("default_response", "A"))
        responses: list[GenerationResponse] = []
        for request in requests:
            started = time.perf_counter()
            text = str(answers.get(request.sample_id, default))
            if text == "__PROMPT_LAST_CHOICE__":
                choices = re.findall(r"\(([A-F])\)", request.prompt.upper())
                text = choices[-1] if choices else default
            responses.append(
                GenerationResponse(
                    sample_id=request.sample_id,
                    text=text,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    metadata={"adapter": "mock"},
                )
            )
        return responses
