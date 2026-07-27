#!/usr/bin/env bash
# Force the shared HMA correctness flow through CUDA Engine-driven transfer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMCACHE_MP_TRANSFER_MODE=engine_driven
export EXPECTED_MP_TRANSFER_MODE=engine_driven
exec "${SCRIPT_DIR}/run-hma-lm-eval.sh"
