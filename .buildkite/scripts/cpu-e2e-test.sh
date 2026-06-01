#!/usr/bin/env bash

set -euo pipefail

SERVER_PID=""
SERVER_READY_TIMEOUT=180
RESPONSE_FILE=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RESPONSE_FILE}" ]]; then
        rm -f "${RESPONSE_FILE}"
    fi
}

trap cleanup EXIT INT TERM

uv venv --python 3.12 .venv-cpu-test
source .venv-cpu-test/bin/activate

uv pip install vllm
uv pip install -r requirements/common.txt && uv pip install -e . --no-build-isolation

python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --device cpu \
    --dtype float32 \
    --max-model-len 512 \
    --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' &
SERVER_PID=$!

echo "Waiting for vLLM server readiness..."
if ! timeout "${SERVER_READY_TIMEOUT}" bash -c 'until curl -s http://localhost:8000/v1/models | grep -q "opt-125m"; do sleep 2; done'; then
    echo "Timed out waiting for server readiness"
    exit 1
fi

echo "Server is ready"

RESPONSE_FILE=$(mktemp /tmp/lmcache_cpu_e2e_response.XXXXXX.json)
http_code=$(curl -s -o "${RESPONSE_FILE}" -w "%{http_code}" \
    http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "facebook/opt-125m",
      "prompt": "The capital of France is",
      "max_tokens": 20,
      "temperature": 0.0
    }')

echo "Response:"
cat "${RESPONSE_FILE}"

if [[ "${http_code}" != "200" ]]; then
    echo
    echo "Expected HTTP 200 but got ${http_code}"
    exit 1
fi

echo
echo "CPU vLLM + LMCache E2E test passed"
