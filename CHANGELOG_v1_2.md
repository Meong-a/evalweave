# EvalWeave v1.2.0

## v1.2.1 loader correction

- Added `common_v1_2.py` so formal JSONL tasks read only physical newline delimiters.
- This preserves valid Unicode line-separator characters embedded in benchmark text.
- The v1/v1.1 loader remains unchanged.
- Added isolated NLTK resource installation and `run_benchmark_v1_2_1.sh` so the official IFEval checker can resolve `punkt_tab` without changing shared environments.
- Added `jsonl_v1_2.py` and switched v1.2 runtime/export readers to physical-line JSONL parsing, preserving Unicode line separators in prompts and model outputs.

- Added full MMLU test evaluation (14,042 items, 57 subjects).
- Added full labelled C-Eval val evaluation (1,346 items) and all unlabelled official test prompts (12,342 items) with submission export.
- Added full BBH evaluation (6,511 items, 27 task files) with official CoT prompts and punctuation-safe target parsing.
- Added full GSM8K test evaluation (1,319 items) with numeric answer extraction.
- Added all 790 TruthfulQA CSV questions under a documented generative MC1 protocol.
- Added all 541 IFEval prompts with the official strict and loose constraint checkers.
- Added all 164 HumanEval tasks with generation export and a Docker-only scorer entrypoint.
- Preserved v1 and v1.1 as separate, unchanged directories.
