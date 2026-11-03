#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIMODAL_PYTHON="${TRIMODAL_PYTHON:-/home/kunyu/miniconda3/envs/trimodal/bin/python}"
exec "${TRIMODAL_PYTHON}" "${SCRIPT_DIR}/run_responses.py" "$@"
