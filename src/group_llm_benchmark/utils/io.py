from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; JSON works without extra dependencies") from exc
        value = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain an object: {path}")
    value["_config_path"] = str(path)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value.strip())
    return cleaned.strip("._") or "run"


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    try:
        import importlib.metadata as metadata

        for name in ("torch", "transformers", "peft", "datasets", "accelerate", "pydantic", "PyYAML"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = "not-installed"
    except Exception as exc:  # pragma: no cover - defensive only
        packages["inspection_error"] = repr(exc)
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, check=False
        ).stdout.strip() or None
    except Exception:
        pass
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": packages,
        "git_commit": git_commit,
    }
