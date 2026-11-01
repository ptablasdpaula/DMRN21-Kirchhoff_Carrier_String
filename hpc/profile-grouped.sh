#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/profile_grouped.py --output "$KC_SCRATCH/profile-grouped-${SLURM_JOB_ID}.json" --batches 1,128 --duration 4 --group 16
