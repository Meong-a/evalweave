#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable


VERSION = "1.2.2"
SEED = 20261021
USER_AGENT = "EvalWeave-Full-Suite/1.2"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"({label}) {text}" for label, text in choices.items())


def mmlu_row(row: list[str], sample_id: str, source: str, few_shot: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "question": row[0],
        "choices": {"A": row[1], "B": row[2], "C": row[3], "D": row[4]},
        "answer": row[5].strip().upper(),
        "source": source,
        "few_shot": few_shot,
    }


def build_mmlu(raw: Path, out: Path) -> tuple[int, Path]:
    data = raw / "mmlu/extracted/data"
    test_files = sorted((data / "test").glob("*_test.csv"))
    if len(test_files) != 57:
        raise ValueError(f"MMLU requires 57 test subjects, got {len(test_files)}")
    output = []
    for test_file in test_files:
        subject = test_file.stem.removesuffix("_test")
        dev_file = data / "dev" / f"{subject}_dev.csv"
        demos = []
        for row in read_csv(dev_file):
            choices = {"A": row[1], "B": row[2], "C": row[3], "D": row[4]}
            demos.append(
                f"Question: {row[0]}\nOptions:\n{format_choices(choices)}\nAnswer: {row[5].strip().upper()}"
            )
        few_shot = "\n\n".join(demos)
        for index, row in enumerate(read_csv(test_file)):
            output.append(mmlu_row(row, f"mmlu:{subject}:{index:05d}", f"mmlu/{subject}", few_shot))
    path = out / "mmlu_test_full_v1_2.jsonl"
    return write_jsonl(path, output), path


def ceval_choices(row: dict[str, str]) -> dict[str, str]:
    return {label: row[label] for label in "ABCD"}


def ceval_demos(dev_file: Path) -> str:
    with dev_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    demos = []
    for row in rows:
        demos.append(
            f"问题：{row['question']}\n选项：\n{format_choices(ceval_choices(row))}\n答案：{row['answer'].strip().upper()}"
        )
    return "\n\n".join(demos)


def build_ceval(raw: Path, out: Path) -> list[tuple[str, int, Path]]:
    data = raw / "ceval/extracted"
    results = []
    for split in ("dev", "val", "test"):
        output = []
        files = sorted((data / split).glob(f"*_{split}.csv"))
        if len(files) != 52:
            raise ValueError(f"C-Eval {split} requires 52 subjects, got {len(files)}")
        for file in files:
            subject = file.stem.removesuffix(f"_{split}")
            few_shot = ceval_demos(data / "dev" / f"{subject}_dev.csv")
            with file.open("r", encoding="utf-8-sig", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    item = {
                        "id": f"ceval:{split}:{subject}:{index:05d}",
                        "original_id": row.get("id", index),
                        "question": row["question"],
                        "choices": ceval_choices(row),
                        "source": f"ceval/{subject}",
                        "few_shot": few_shot,
                    }
                    if split != "test":
                        item["answer"] = row["answer"].strip().upper()
                    output.append(item)
        path = out / f"ceval_{split}_full_v1_2.jsonl"
        results.append((f"ceval_{split}", write_jsonl(path, output), path))
    return results


def build_bbh(raw: Path, out: Path) -> tuple[int, Path]:
    task_root = raw / "bbh/extracted/BIG-Bench-Hard-main/bbh"
    prompt_root = raw / "bbh/cot-prompts"
    prompt_root.mkdir(parents=True, exist_ok=True)
    output = []
    for task in BBH_TASKS:
        prompt_path = prompt_root / f"{task}.txt"
        download(
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{task}.txt",
            prompt_path,
        )
        cot = prompt_path.read_text(encoding="utf-8").strip()
        payload = json.loads((task_root / f"{task}.json").read_text(encoding="utf-8"))
        for index, item in enumerate(payload["examples"]):
            output.append({
                "id": f"bbh:{task}:{index:05d}",
                "prompt": f"{cot}\n\nQ: {item['input']}\nA: Let's think step by step.",
                "input": item["input"],
                "reference": item["target"],
                "source": f"bbh/{task}",
            })
    path = out / "bbh_full_cot_v1_2.jsonl"
    return write_jsonl(path, output), path


def build_gsm8k(raw: Path, out: Path) -> tuple[int, Path]:
    train = raw / "gsm8k/train.jsonl"
    test = raw / "gsm8k/test.jsonl"
    download(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        train,
    )
    train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    demos = []
    for row in train_rows[:8]:
        demos.append(f"Question: {row['question']}\nAnswer: {row['answer']}")
    few_shot = "\n\n".join(demos)
    output = []
    for index, line in enumerate(test.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        output.append({
            "id": f"gsm8k:test:{index:05d}",
            "input": row["question"],
            "reference": row["answer"].rsplit("####", 1)[-1].strip().replace(",", ""),
            "solution": row["answer"],
            "few_shot": few_shot,
            "source": "gsm8k/test",
        })
    path = out / "gsm8k_test_full_v1_2.jsonl"
    return write_jsonl(path, output), path


def parse_answer_list(value: str) -> list[str]:
    value = (value or "").strip()
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [item.strip() for item in value.split(";") if item.strip()]


def build_truthfulqa(raw: Path, out: Path) -> tuple[int, Path]:
    source = raw / "truthfulqa/TruthfulQA.csv"
    output = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            best = row["Best Answer"].strip()
            incorrect = parse_answer_list(row["Incorrect Answers"])
            if not best or not incorrect:
                raise ValueError(f"TruthfulQA row {index} lacks MC1 choices")
            answers = [best] + incorrect
            if len(answers) > len(LABELS):
                raise ValueError(f"TruthfulQA row {index} has {len(answers)} choices")
            rng = random.Random(f"{SEED}:truthfulqa:{index}")
            rng.shuffle(answers)
            choices = {label: answer for label, answer in zip(LABELS, answers)}
            correct = next(label for label, answer in choices.items() if answer == best)
            output.append({
                "id": f"truthfulqa:mc1:{index:04d}",
                "question": row["Question"],
                "choices": choices,
                "answer": correct,
                "source": f"truthfulqa/{row.get('Category', 'unknown')}",
                "metadata": {
                    "category": row.get("Category", "unknown"),
                    "protocol": "generative_mc1_best_answer_vs_all_incorrect_deterministic_shuffle",
                },
            })
    path = out / "truthfulqa_mc1_full_v1_2.jsonl"
    return write_jsonl(path, output), path


def build_ifeval(raw: Path, out: Path) -> tuple[int, Path]:
    source = raw / "ifeval/input_data.jsonl"
    output = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        row["id"] = f"ifeval:{row.get('key', index)}"
        row["source"] = "ifeval/official"
        output.append(row)
    path = out / "ifeval_full_v1_2.jsonl"
    return write_jsonl(path, output), path


def build_humaneval(raw: Path, out: Path) -> tuple[int, Path]:
    source = raw / "humaneval/HumanEval.jsonl.gz"
    output = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            row["id"] = f"humaneval:{row['task_id']}"
            row["source"] = "humaneval/official"
            output.append(row)
    path = out / "humaneval_full_v1_2.jsonl"
    return write_jsonl(path, output), path


def write_task_configs(root: Path) -> None:
    config_root = root / "configs/tasks_v1_2"
    config_root.mkdir(parents=True, exist_ok=True)
    data = root / "data/full_v1_2"
    common_mcq = {
        "type": "mcq_formal", "id_field": "id", "question_field": "question",
        "choices_field": "choices", "answer_field": "answer", "source_field": "source",
        "primary_metric": "accuracy",
    }
    specs = {
        "mmlu_test_full_v1_2.json": {**common_mcq, "name": "mmlu_test_full_v1_2",
            "dataset_path": str(data / "mmlu_test_full_v1_2.jsonl"),
            "capabilities": ["english_knowledge", "domain_knowledge"],
            "prompt_template": "{few_shot}\n\nQuestion: {question}\nOptions:\n{choices}\nAnswer with only the option letter:"},
        "ceval_val_full_v1_2.json": {**common_mcq, "name": "ceval_val_full_v1_2",
            "dataset_path": str(data / "ceval_val_full_v1_2.jsonl"),
            "capabilities": ["chinese_knowledge", "domain_knowledge"],
            "prompt_template": "{few_shot}\n\n问题：{question}\n选项：\n{choices}\n只回答选项字母："},
        "ceval_test_submission_full_v1_2.json": {"type": "mcq_submission",
            "name": "ceval_test_submission_full_v1_2",
            "dataset_path": str(data / "ceval_test_full_v1_2.jsonl"),
            "capabilities": ["chinese_knowledge", "domain_knowledge"],
            "primary_metric": "generation_success",
            "prompt_template": "{few_shot}\n\n问题：{question}\n选项：\n{choices}\n只回答选项字母："},
        "bbh_full_cot_v1_2.json": {"type": "bbh", "name": "bbh_full_cot_v1_2",
            "dataset_path": str(data / "bbh_full_cot_v1_2.jsonl"),
            "capabilities": ["complex_reasoning"], "primary_metric": "exact_match"},
        "gsm8k_test_full_v1_2.json": {"type": "numeric_qa", "name": "gsm8k_test_full_v1_2",
            "dataset_path": str(data / "gsm8k_test_full_v1_2.jsonl"),
            "capabilities": ["mathematical_reasoning"], "primary_metric": "numeric_exact_match",
            "prompt_template": "{few_shot}\n\nQuestion: {text}\nAnswer: Let's think step by step."},
        "truthfulqa_mc1_full_v1_2.json": {**common_mcq, "name": "truthfulqa_mc1_full_v1_2",
            "dataset_path": str(data / "truthfulqa_mc1_full_v1_2.jsonl"),
            "capabilities": ["truthfulness", "misconception_resistance"],
            "prompt_template": "Select the most truthful answer. Reply with only the option letter.\nQuestion: {question}\nOptions:\n{choices}\nAnswer:"},
        "ifeval_full_v1_2.json": {"type": "ifeval", "name": "ifeval_full_v1_2",
            "dataset_path": str(data / "ifeval_full_v1_2.jsonl"),
            "capabilities": ["instruction_following"], "primary_metric": "strict_prompt_accuracy"},
        "humaneval_full_v1_2.json": {"type": "code_generation", "name": "humaneval_full_v1_2",
            "dataset_path": str(data / "humaneval_full_v1_2.jsonl"),
            "capabilities": ["code_generation"], "primary_metric": "generation_success",
            "requires_postprocess": "scripts/export_humaneval_samples_v1_2.py",
            "requires_sandbox_scorer": "scripts/score_humaneval_docker_v1_2.sh"},
    }
    for filename, payload in specs.items():
        (config_root / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-raw", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data/raw_v1_2"
    if not raw.exists():
        shutil.copytree(args.source_raw.resolve(), raw)
    out = root / "data/full_v1_2"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    builders = [
        ("mmlu_test", lambda: build_mmlu(raw, out)),
        ("bbh", lambda: build_bbh(raw, out)),
        ("gsm8k_test", lambda: build_gsm8k(raw, out)),
        ("truthfulqa", lambda: build_truthfulqa(raw, out)),
        ("ifeval", lambda: build_ifeval(raw, out)),
        ("humaneval", lambda: build_humaneval(raw, out)),
    ]
    for name, builder in builders:
        count, path = builder()
        results[name] = {"count": count, "path": str(path), "sha256": sha256(path)}
        print(f"built {name}: {count}")
    for name, count, path in build_ceval(raw, out):
        results[name] = {"count": count, "path": str(path), "sha256": sha256(path)}
        print(f"built {name}: {count}")

    write_task_configs(root)
    manifest = {
        "suite": "EvalWeave Full Suite",
        "version": VERSION,
        "seed": SEED,
        "results": results,
        "scored_evaluation_total": 24713,
        "normalized_records_total_including_ceval_dev_and_unlabelled_test": 37315,
    }
    manifest_path = root / "manifests/build_manifest_v1_2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
