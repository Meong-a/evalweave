#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/humaneval_samples.jsonl /absolute/output_directory" >&2
  exit 2
fi
SAMPLES="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
if [[ ! -f "${SAMPLES}" ]]; then
  echo "Samples file not found: ${SAMPLES}" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable to the current user; refusing unsafe host execution." >&2
  exit 3
fi
mkdir -p "${OUTPUT}"
cp "${SAMPLES}" "${OUTPUT}/samples.jsonl"
docker build -t evalweave-humaneval:v1.2 -f "${ROOT}/docker/humaneval_v1_2/Dockerfile" "${ROOT}"
docker run --rm \
  --network none \
  --memory 2g \
  --cpus 2 \
  --pids-limit 128 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=512m \
  -v "${OUTPUT}:/work:rw" \
  evalweave-humaneval:v1.2 /work/samples.jsonl

