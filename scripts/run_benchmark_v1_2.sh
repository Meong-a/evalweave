#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/share/anaconda/envs/ms-swift/bin/python}"
export PYTHONPATH="${ROOT}/src:${ROOT}/.deps_ifeval_v1_2${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m group_llm_benchmark "$@"

