#!/usr/bin/env bash
set -euo pipefail

echo "Build ID: ${BUILDKITE_BUILD_ID:-local}"
echo "Python: $(python3 --version 2>&1 || true)"
echo "uv: $(uv --version 2>&1 || true)"

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
VENV_DIR=".venv-${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache_cpu_validation.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm_cpu_validation.log"
LMCACHE_PID=""
VLLM_PID=""
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-2}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-30}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-120}"

cleanup_workspace() {
  if [ -n "${BUILDKITE_BUILD_ID:-}" ]; then
    export TARGET="$PWD"
    case "$TARGET" in
      ""|"/"|"/usr"|"/var"|"/etc"|"/bin"|"/sbin"|"/opt"|"/home"|"/tmp")
        echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
        return 1
        ;;
    esac
    if [ "$TARGET" = "$HOME" ]; then
      echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
      return 1
    fi
    if [ ! -d "$TARGET/.git" ] || [ ! -f "$TARGET/pyproject.toml" ]; then
      echo "❌ Refusing to delete unexpected workspace path: $TARGET"
      return 1
    fi
    echo "Deleting current workspace $TARGET"
    cd /
    if command -v sudo >/dev/null 2>&1; then
      sudo rm -rf "$TARGET"
    else
      rm -rf "$TARGET"
    fi
  fi
}

print_failure_logs() {
  echo "=== LMCache Server Log (${LMCACHE_LOG}) ==="
  if [ -f "${LMCACHE_LOG}" ]; then
    tail -n 200 "${LMCACHE_LOG}" || true
  else
    echo "Log not found"
  fi
  echo "=== vLLM Log (${VLLM_LOG}) ==="
  if [ -f "${VLLM_LOG}" ]; then
    tail -n 200 "${VLLM_LOG}" || true
  else
    echo "Log not found"
  fi
}

cleanup_processes() {
  set +e
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [ -n "${LMCACHE_PID}" ] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    echo "Stopping LMCache server (PID=${LMCACHE_PID})"
    kill "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
  fi
  set -e
}

wait_for_endpoint_contains() {
  local url="$1"
  local timeout="$2"
  local expected="$3"
  local label="$4"
  local response

  for _ in $(seq 1 "${timeout}"); do
    if response="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [ -z "${expected}" ] || echo "${response}" | grep -q "${expected}"; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "❌ ${label} did not become ready within ${timeout}s"
  return 1
}

on_error() {
  local exit_code=$?
  trap - ERR
  echo "❌ CPU install validation failed (exit code: ${exit_code})"
  set +e
  print_failure_logs
  cleanup_processes
  cleanup_workspace || echo "❌ Workspace cleanup failed"
  set -e
  exit "$exit_code"
}

trap on_error ERR

echo "=== CPU Install Validation (Phase 1) ==="
echo "Creating virtual environment with uv at ${VENV_DIR}"
uv venv --python 3.12 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo "✅ Virtual environment ready"

echo "Upgrading pip/setuptools/wheel"
uv pip install --upgrade pip setuptools wheel
echo "✅ Upgraded pip/setuptools/wheel"

echo "Installing build dependencies from requirements/build.txt"
uv pip install -r requirements/build.txt
echo "✅ Installed requirements/build.txt"

echo "Installing common dependencies from requirements/common.txt"
uv pip install -r requirements/common.txt
echo "✅ Installed requirements/common.txt"

echo "Installing vLLM CPU build"
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
echo "✅ vLLM CPU install completed"

echo "Installing LMCache in editable mode with NO_GPU_EXT=1"
NO_GPU_EXT=1 uv pip install -e . --no-build-isolation
echo "✅ LMCache install completed"

echo "Freezing installed package versions"
uv pip freeze

echo "Validating imports"
python -c "import lmcache; import vllm; print('✅ Imports OK')"

echo "Printing package versions"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import lmcache; print('lmcache:', lmcache.__version__)"

echo "✅ CPU install validation passed"

echo "=== CPU E2E Validation (Phase 2) ==="

echo "[Phase 2 / Step 1] Installing numpy<2 for scipy/vLLM compatibility"
uv pip install "numpy<2"
echo "✅ numpy<2 installed"

echo "[Phase 2 / Step 2] Downloading facebook/opt-125m model (cache-aware)"
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')"; then
  echo "❌ Failed to download/cache facebook/opt-125m"
  false
fi
echo "✅ Model download/check complete"

echo "[Phase 2 / Step 3] Starting LMCache server"
echo "LMCache log: ${LMCACHE_LOG}"
lmcache server \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --eviction-policy "${LMCACHE_EVICTION_POLICY}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
echo "LMCache server started (PID=${LMCACHE_PID})"
sleep 1
if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
  echo "❌ LMCache server exited immediately after startup. See ${LMCACHE_LOG} for details"
  false
fi

echo "Waiting for LMCache healthcheck at http://localhost:${LMCACHE_HTTP_PORT}/healthcheck (timeout: ${LMCACHE_HEALTHCHECK_TIMEOUT}s)"
if ! wait_for_endpoint_contains "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" "${LMCACHE_HEALTHCHECK_TIMEOUT}" "" "LMCache server"; then
  false
fi
echo "✅ LMCache server is healthy"

echo "[Phase 2 / Step 4] Starting vLLM server"
echo "vLLM log: ${VLLM_LOG}"
apt-get update && apt-get install -y --no-install-recommends libnuma1
export VLLM_TARGET_DEVICE=cpu
VLLM_TARGET_DEVICE=cpu vllm serve facebook/opt-125m \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM server started (PID=${VLLM_PID})"
sleep 1
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "❌ vLLM server exited immediately after startup. See ${VLLM_LOG} for details"
  false
fi

echo "Waiting for vLLM readiness at http://localhost:${VLLM_PORT}/v1/models (timeout: ${VLLM_READY_TIMEOUT}s)"
if ! wait_for_endpoint_contains "http://localhost:${VLLM_PORT}/v1/models" "${VLLM_READY_TIMEOUT}" "facebook/opt-125m" "vLLM server"; then
  false
fi
echo "✅ vLLM server is ready"

echo "[Phase 2 / Step 5] Sending E2E test request"
completion_response="$(curl -fsS "http://localhost:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"facebook/opt-125m","prompt":"Hello","max_tokens":5}')"
echo "Completion response: ${completion_response}"
if ! echo "${completion_response}" | grep -q '"choices"'; then
  echo "❌ E2E request response failed structural validation"
  false
fi
if ! echo "${completion_response}" | grep -q "facebook/opt-125m"; then
  echo "❌ E2E request response missing expected model"
  false
fi
echo "✅ E2E request validation passed"

echo "[Phase 2 / Step 6] Cleaning up LMCache and vLLM processes"
cleanup_processes
echo "✅ Phase 2 cleanup completed"

echo "✅ CPU E2E validation passed"
cleanup_workspace
