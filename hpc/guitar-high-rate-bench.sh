#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/guitar_high_rate_benchmark.py --output "$KC_SCRATCH/guitar-high-rate-bench-${SLURM_JOB_ID}.json"
