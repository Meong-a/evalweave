#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/share/anaconda/envs/ms-swift/bin/python}"
TARGET="${ROOT}/.deps_ifeval_v1_2"
mkdir -p "${TARGET}"
"${PYTHON_BIN}" -m pip install --target "${TARGET}" -r "${ROOT}/requirements_ifeval_v1_2.txt"
echo "Installed isolated IFEval dependencies into ${TARGET}"

