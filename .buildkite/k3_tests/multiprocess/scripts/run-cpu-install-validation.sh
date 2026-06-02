#!/usr/bin/env bash
set -e

echo "Build ID: ${BUILDKITE_BUILD_ID:-local}"
echo "Python: $(python3 --version 2>&1 || true)"
echo "uv: $(uv --version 2>&1 || true)"

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
VENV_DIR=".venv-${BUILD_ID}"

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

on_error() {
  local exit_code=$?
  trap - ERR
  echo "❌ CPU install validation failed (exit code: ${exit_code})"
  set +e
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
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu --prerelease=allow
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
cleanup_workspace
