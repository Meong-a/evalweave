from __future__ import annotations

from typing import Any

from .base import ModelAdapter
from .mock import MockAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .transformers import TransformersAdapter


def create_adapter(config: dict[str, Any]) -> ModelAdapter:
    adapter_type = str(config.get("type", "")).lower()
    adapters: dict[str, type[ModelAdapter]] = {
        "mock": MockAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
        "transformers": TransformersAdapter,
        "peft": TransformersAdapter,
        "lora": TransformersAdapter,
    }
    try:
        return adapters[adapter_type](config)
    except KeyError as exc:
        raise ValueError(f"Unknown model adapter type: {adapter_type}") from exc
