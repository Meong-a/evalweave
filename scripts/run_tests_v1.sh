#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/share/anaconda/envs/ms-swift/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${PYTHON_BIN}" -m unittest discover -s "${PROJECT_ROOT}/tests" -p "test_*.py" -v
