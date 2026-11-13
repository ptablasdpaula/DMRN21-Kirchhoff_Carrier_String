#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cores="$($KC_PYTHON -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')"
taskset -c "$cores" "$KC_SCRATCH/jax-venv/bin/python" benchmarks/jax_reverse_compare.py --output "$KC_SCRATCH/jax-reverse-${SLURM_JOB_ID}.json" --durations .025,.25,4 --batches 1,8 --max-predicted-seconds 20 --repeats 3
