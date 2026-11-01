#!/bin/bash
set -euo pipefail
source hpc/environment.sh
# All nine template widths are compiled in this extension; bound compiler memory.
export MAX_JOBS=1
"$KC_PYTHON" test/experimental_kernels.py --output "$KC_SCRATCH/experimental-kernels-$SLURM_JOB_ID.json"
"$KC_PYTHON" test/seeded.py --device cuda
"$KC_PYTHON" test/seeded.py --device cpu
