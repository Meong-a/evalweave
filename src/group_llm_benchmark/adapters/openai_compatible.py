from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from group_llm_benchmark.schemas import GenerationRequest, GenerationResponse

from .base import ModelAdapter


class OpenAICompatibleAdapter(ModelAdapter):
    """Minimal OpenAI-compatible Chat Completions client without SDK dependency."""

    def generate(
        self, requests: Sequence[GenerationRequest], generation_config: dict[str, Any]
    ) -> list[GenerationResponse]:
        base_url = str(self.config.get("base_url", "")).rstrip("/")
        model = str(self.config.get("model", ""))
        if not base_url or not model:
            raise ValueError("openai_compatible adapter requires base_url and model")
        api_key_env = str(self.config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env, "")
        timeout = float(self.config.get("timeout_seconds", 120))
        retries = int(self.config.get("retries", 2))
        results: list[GenerationResponse] = []
        for request in requests:
            started = time.perf_counter()
            messages = list(request.messages) or [{"role": "user", "content": request.prompt}]
            payload = {"model": model, "messages": messages, **generation_config}
            body = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            error: str | None = None
            text = ""
            for attempt in range(retries + 1):
                try:
                    req = urllib.request.Request(
                        f"{base_url}/chat/completions", data=body, headers=headers, method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    text = str(data["choices"][0]["message"]["content"])
                    error = None
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt >= retries:
                        break
                    time.sleep(min(2**attempt, 8))
            results.append(
                GenerationResponse(
                    sample_id=request.sample_id,
                    text=text,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=error,
                    metadata={"adapter": "openai_compatible", "model": model},
                )
            )
        return results
