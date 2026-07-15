#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tiny-dk}"

curl "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"prompt\": \"def add(a, b):\",
    \"max_tokens\": 16,
    \"temperature\": 0
  }"
