#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


USER_AGENT = "EvalWeave/1.2.2"
SOURCES = {
    "mmlu": "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
    "ceval": "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip",
    "bbh": "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "truthfulqa": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "ifeval": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
    "humaneval": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        print(f"reuse {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"download {url}")
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe tar path: {member.name}")
        bundle.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            target = (destination / name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe zip path: {name}")
        bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official sources for the full EvalWeave suite")
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data/raw_v1_2"

    mmlu_archive = raw / "mmlu/data.tar"
    download(SOURCES["mmlu"], mmlu_archive)
    if not (raw / "mmlu/extracted/data/test").exists():
        safe_extract_tar(mmlu_archive, raw / "mmlu/extracted")

    ceval_archive = raw / "ceval/ceval-exam.zip"
    download(SOURCES["ceval"], ceval_archive)
    if not (raw / "ceval/extracted/val").exists():
        safe_extract_zip(ceval_archive, raw / "ceval/extracted")

    bbh_archive = raw / "bbh/bbh-main.zip"
    download(SOURCES["bbh"], bbh_archive)
    if not (raw / "bbh/extracted/BIG-Bench-Hard-main/bbh").exists():
        safe_extract_zip(bbh_archive, raw / "bbh/extracted")

    download(SOURCES["gsm8k"], raw / "gsm8k/test.jsonl")
    download(SOURCES["truthfulqa"], raw / "truthfulqa/TruthfulQA.csv")
    download(SOURCES["ifeval"], raw / "ifeval/input_data.jsonl")
    download(SOURCES["humaneval"], raw / "humaneval/HumanEval.jsonl.gz")

    manifest = {"suite_version": "1.2.2", "sources": SOURCES}
    destination = root / "manifests/source_urls_v1_2_2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"official source preparation complete: {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
