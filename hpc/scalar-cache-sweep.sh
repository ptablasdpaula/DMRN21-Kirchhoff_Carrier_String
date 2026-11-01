#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
"$KC_PYTHON" benchmarks/scalar_cache_sweep.py --output "$KC_SCRATCH/scalar-cache-sweep-${SLURM_JOB_ID}.json" --durations .25,4 --batches 1,8,128 --directions 2,6 --groups 0,16 --checkpoint 256 --repeats 3
