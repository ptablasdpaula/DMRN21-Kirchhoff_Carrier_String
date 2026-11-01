#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/cache_reconstruction_compare.py --output "$KC_SCRATCH/cache-reconstruction-compare-${SLURM_JOB_ID}.json" --durations .25,4 --batches 1,128 --checkpoint 256 --repeats 3
