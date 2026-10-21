#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                raise ValueError(f"failed prediction: {row['sample_id']}")
            prediction = row.get("parsed_prediction")
            if prediction is None:
                raise ValueError(f"invalid prediction: {row['sample_id']}")
            source = str(row["metadata"]["source"])
            subject = source.split("/", 1)[-1]
            original_id = str(row["metadata"]["original_id"])
            grouped[subject][original_id] = str(prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(grouped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
