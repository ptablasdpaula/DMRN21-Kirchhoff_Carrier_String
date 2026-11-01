#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" test/alpha_scan_coordinate_stress.py --output "$KC_SCRATCH/alpha-scan-coordinates-${SLURM_JOB_ID}.json"
