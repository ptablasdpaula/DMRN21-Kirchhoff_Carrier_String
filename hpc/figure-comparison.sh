#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
CASE_PYTHON="$KC_SCRATCH/jax-venv/bin/python"
"$CASE_PYTHON" benchmarks/figure_comparison_driver.py --output "$KC_SCRATCH/figure-comparison-${SLURM_JOB_ID}.json" --plot-prefix "$KC_SCRATCH/figure-comparison-${SLURM_JOB_ID}" --python "$CASE_PYTHON"
"$KC_PYTHON" benchmarks/figure_comparison_plot.py "$KC_SCRATCH/figure-comparison-${SLURM_JOB_ID}.json" --output-prefix "$KC_SCRATCH/figure-comparison-${SLURM_JOB_ID}"
