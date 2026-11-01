#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/cache_grouped_compare.py --output "$KC_SCRATCH/cache-grouped-compare-${SLURM_JOB_ID}.json" --batch 128 --duration 4 --group 16 --repeats 3
