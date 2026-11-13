#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" test/guitar_high_rate_smoke.py --oversampling "${1:-55}" --output "$KC_SCRATCH/guitar-high-rate-${SLURM_JOB_ID}.json"
