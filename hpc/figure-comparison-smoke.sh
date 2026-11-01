#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
"$KC_PYTHON" test/figure_fdtd.py --cuda
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/figure_comparison_driver.py --output "$KC_SCRATCH/figure-comparison-smoke-${SLURM_JOB_ID}.json" --batches 1 --methods cached3,cached1,diaz,fdtd --duration .03 --timeout 600 --excitation raised_cosine --python "$KC_SCRATCH/jax-venv/bin/python"
