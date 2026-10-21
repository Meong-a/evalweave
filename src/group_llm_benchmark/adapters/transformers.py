from __future__ import annotations

import time
from typing import Any, Sequence

from group_llm_benchmark.schemas import GenerationRequest, GenerationResponse

from .base import ModelAdapter


class TransformersAdapter(ModelAdapter):
    """Hugging Face causal-LM adapter with optional PEFT/LoRA loading."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers adapter requires torch and transformers") from exc
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("transformers adapter requires model_path")
        trust_remote_code = bool(self.config.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            padding_side=str(self.config.get("padding_side", "left")),
            local_files_only=bool(self.config.get("local_files_only", True)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_name = str(self.config.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        model_kwargs = dict(self.config.get("model_kwargs", {}))
        device = str(self.config.get("device", "cuda:0"))
        if device.startswith("cuda"):
            device_index = int(device.split(":", 1)[1]) if ":" in device else 0
            model_kwargs.setdefault("device_map", {"": device_index})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            local_files_only=bool(self.config.get("local_files_only", True)),
            **model_kwargs,
        )
        adapter_path = self.config.get("adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("adapter_path requires peft") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=True)
        self.model.eval()

    def generate(
        self, requests: Sequence[GenerationRequest], generation_config: dict[str, Any]
    ) -> list[GenerationResponse]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("adapter is not loaded")
        import torch

        prompts = [request.prompt for request in requests]
        max_input_length = int(self.config.get("max_input_length", 2048))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs = dict(generation_config)
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("max_new_tokens", 64)
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        input_width = encoded["input_ids"].shape[1]
        new_tokens = generated[:, input_width:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        each_latency = elapsed_ms / max(len(requests), 1)
        return [
            GenerationResponse(
                sample_id=request.sample_id,
                text=text,
                latency_ms=each_latency,
                metadata={"adapter": "transformers"},
            )
            for request, text in zip(requests, texts)
        ]

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
