#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/share/anaconda/envs/ms-swift/bin/python}"
TARGET="${ROOT}/.nltk_data_ifeval_v1_2"
mkdir -p "${TARGET}"
PYTHONPATH="${ROOT}/.deps_ifeval_v1_2${PYTHONPATH:+:${PYTHONPATH}}" \
  NLTK_DATA="${TARGET}" \
  "${PYTHON_BIN}" -m nltk.downloader -d "${TARGET}" punkt_tab
echo "Installed isolated IFEval NLTK data into ${TARGET}"
