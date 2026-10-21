from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL by physical lines, preserving Unicode line separators in text."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", newline="") as handle:
            values = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        values = loaded if isinstance(loaded, list) else loaded.get("data")
    else:
        raise ValueError(f"Task data must be .json or .jsonl: {path}")
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise ValueError(f"Task data must contain a list of objects: {path}")
    return values


def format_choices(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"({key}) {text}" for key, text in value.items())
    if isinstance(value, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"({labels[index]}) {text}" for index, text in enumerate(value))
    raise ValueError(f"Unsupported choices value: {type(value).__name__}")
