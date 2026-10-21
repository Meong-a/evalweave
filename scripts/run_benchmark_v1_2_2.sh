#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT}/src:${ROOT}/.deps_ifeval_v1_2${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA="${ROOT}/.nltk_data_ifeval_v1_2${NLTK_DATA:+:${NLTK_DATA}}"

"${PYTHON_BIN}" -m group_llm_benchmark "$@"
