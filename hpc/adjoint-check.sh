#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
"$KC_PYTHON" test/adjoint_math.py
"$KC_PYTHON" test/adjoint.py
