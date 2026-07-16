#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/tiny-dk-a10}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tiny-dk}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
DTYPE="${DTYPE:-float16}"

uv run --no-project tools/create_tiny_dk_model_dir.py \
  --output-dir "${MODEL_DIR}" \
  --force

VLLM_USE_V1="${VLLM_USE_V1:-1}" uv run --no-project vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --load-format dummy \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager
