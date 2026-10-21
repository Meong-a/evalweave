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
IFEVAL_NAMES = ["evaluation_lib.py", "instructions.py", "instructions_registry.py", "instructions_util.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EvalWeave/1.2.1"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    package = root / "src/instruction_following_eval"
    package.mkdir(parents=True, exist_ok=True)
    init = package / "__init__.py"
    init.write_text(
        '"""Vendored official Google Research IFEval scorer package."""\n',
        encoding="utf-8",
    )
    records = [{"project": "EvalWeave", "path": str(init), "sha256": sha256(init)}]
    for name in IFEVAL_NAMES:
        target = package / name
        download(f"{GOOGLE_RAW}/instruction_following_eval/{name}", target)
        records.append({"project": "google-research/IFEval", "path": str(target), "sha256": sha256(target)})
    license_path = root / "third_party/licenses/google_research_LICENSE"
    download(f"{GOOGLE_RAW}/LICENSE", license_path)

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
    manifest.write_text(
        json.dumps({"suite_version": "1.2.0", "fetch_patch": "1.2.1", "sources": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

