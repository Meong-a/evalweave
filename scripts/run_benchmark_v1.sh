#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${PYTHON_BIN}" -m group_llm_benchmark "$@"
