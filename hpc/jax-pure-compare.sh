#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cores="$($KC_PYTHON -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')"
taskset -c "$cores" "$KC_SCRATCH/jax-venv/bin/python" benchmarks/jax_pure_compare.py --output "$KC_SCRATCH/jax-pure-${SLURM_JOB_ID}.json" --durations .025,.25,4 --batches 1,8 --directions 2,6 --max-predicted-seconds 20 --repeats 3
