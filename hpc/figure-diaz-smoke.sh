#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/figure_comparison_driver.py --output "$KC_SCRATCH/figure-diaz-smoke-${SLURM_JOB_ID}.json" --batches 1,2 --methods diaz --duration .03 --timeout 600 --python "$KC_SCRATCH/jax-venv/bin/python"
