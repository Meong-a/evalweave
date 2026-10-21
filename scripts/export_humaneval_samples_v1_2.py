#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                raise ValueError(f"failed prediction: {row['sample_id']}")
            completion = row.get("parsed_prediction")
            if completion is None:
                raise ValueError(f"empty completion: {row['sample_id']}")
            rows.append({"task_id": row["reference"]["task_id"], "completion": strip_fence(str(completion))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
