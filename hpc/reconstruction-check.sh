#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
"$KC_PYTHON" test/reconstruction_math.py
"$KC_PYTHON" test/reconstruction.py --output "$KC_SCRATCH/reconstruction-validation-${SLURM_JOB_ID}.json"
