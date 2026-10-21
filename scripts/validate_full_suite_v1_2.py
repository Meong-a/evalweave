#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "mmlu_test": ("mmlu_test_full_v1_2.jsonl", 14042, "mcq"),
    "ceval_dev": ("ceval_dev_full_v1_2.jsonl", 260, "mcq"),
    "ceval_val": ("ceval_val_full_v1_2.jsonl", 1346, "mcq"),
    "ceval_test": ("ceval_test_full_v1_2.jsonl", 12342, "submission"),
    "bbh": ("bbh_full_cot_v1_2.jsonl", 6511, "bbh"),
    "gsm8k_test": ("gsm8k_test_full_v1_2.jsonl", 1319, "numeric"),
    "truthfulqa": ("truthfulqa_mc1_full_v1_2.jsonl", 790, "mcq"),
    "ifeval": ("ifeval_full_v1_2.jsonl", 541, "ifeval"),
    "humaneval": ("humaneval_full_v1_2.jsonl", 164, "humaneval"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_row(kind: str, row: dict[str, Any], location: str) -> None:
    required = {
        "mcq": {"id", "question", "choices", "answer", "source"},
        "submission": {"id", "question", "choices", "source", "original_id"},
        "bbh": {"id", "prompt", "reference", "source"},
        "numeric": {"id", "input", "reference", "source"},
        "ifeval": {"id", "key", "prompt", "instruction_id_list", "kwargs", "source"},
        "humaneval": {"id", "task_id", "prompt", "test", "entry_point", "source"},
    }[kind]
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{location}: missing {sorted(missing)}")
    if kind == "mcq":
        if not isinstance(row["choices"], dict) or row["answer"] not in row["choices"]:
            raise ValueError(f"{location}: answer does not identify a choice")
    if kind == "submission" and "answer" in row:
        raise ValueError(f"{location}: unlabelled test row unexpectedly contains answer")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifests/build_manifest_v1_2.json").read_text(encoding="utf-8"))
    all_ids: set[str] = set()
    summary: dict[str, Any] = {}
    total = 0
    for name, (filename, expected_count, kind) in EXPECTED.items():
        path = root / "data/full_v1_2" / filename
        count = 0
        sources: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                validate_row(kind, row, f"{path}:{line_number}")
                sample_id = str(row["id"])
                if sample_id in all_ids:
                    raise ValueError(f"duplicate sample id: {sample_id}")
                all_ids.add(sample_id)
                sources.add(str(row["source"]))
                count += 1
        if count != expected_count:
            raise ValueError(f"{name}: expected {expected_count}, got {count}")
        recorded = manifest["results"][name]
        digest = sha256(path)
        if digest != recorded["sha256"] or count != recorded["count"]:
            raise ValueError(f"{name}: manifest mismatch")
        summary[name] = {"count": count, "sources": len(sources), "sha256": digest}
        total += count
    if total != 37315:
        raise ValueError(f"expected 37315 normalized records, got {total}")
    report = {
        "status": "PASS",
        "suite_version": "1.2.2",
        "build_scope": "FULL",
        "normalized_records_total": total,
        "scored_evaluation_total": 24713,
        "ceval_unlabelled_test_total": 12342,
        "datasets": summary,
    }
    destination = root / "manifests/validation_report_v1_2.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
