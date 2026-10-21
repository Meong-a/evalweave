#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


GOOGLE_RAW = "https://raw.githubusercontent.com/google-research/google-research/master"
IFEVAL_FILES = [
    "instruction_following_eval/__init__.py",
    "instruction_following_eval/evaluation_lib.py",
    "instruction_following_eval/instructions.py",
    "instruction_following_eval/instructions_registry.py",
    "instruction_following_eval/instructions_util.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EvalWeave/1.2"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for relative in IFEVAL_FILES:
        target = root / "src" / Path(relative).relative_to("instruction_following_eval").parent
        target = root / "src/instruction_following_eval" / Path(relative).name
        download(f"{GOOGLE_RAW}/{relative}", target)
        records.append({"project": "google-research/IFEval", "path": str(target), "sha256": sha256(target)})
    download(f"{GOOGLE_RAW}/LICENSE", root / "third_party/licenses/google_research_LICENSE")

    human_eval = root / "third_party/human-eval"
    if not human_eval.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/openai/human-eval.git", str(human_eval)],
            check=True,
        )
    commit = subprocess.check_output(["git", "-C", str(human_eval), "rev-parse", "HEAD"], text=True).strip()
    records.append({"project": "openai/human-eval", "path": str(human_eval), "commit": commit})
    manifest = root / "manifests/official_evaluators_manifest_v1_2.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"version": "1.2.0", "sources": records}, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

