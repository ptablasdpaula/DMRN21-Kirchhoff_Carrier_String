#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
CASE_METHODS="$1"
CASE_BATCHES="$2"
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/figure_comparison_driver.py \
  --output "$KC_SCRATCH/figure-shard-${SLURM_JOB_ID}.json" \
  --methods "$CASE_METHODS" --batches "$CASE_BATCHES" --duration 4 \
  --timeout 1800 --python "$KC_SCRATCH/jax-venv/bin/python"
