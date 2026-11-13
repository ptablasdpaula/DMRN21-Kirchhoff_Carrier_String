#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" test/manual_ce_long.py --output "$KC_SCRATCH/manual-ce-long-${SLURM_JOB_ID}.json" --device cuda
