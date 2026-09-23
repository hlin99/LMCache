#!/usr/bin/env bash
set -euo pipefail

bash scripts/install_xpu_dpcpp_compiler.sh

set +u
source /opt/intel/oneapi/setvars.sh
set -u

python -m pip install -r requirements/build.txt

BUILD_WITH_SYCL=1 python -m pip install --no-build-isolation -e .
