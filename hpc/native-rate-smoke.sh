#!/bin/bash
# Short GPU smoke of the one-second raised-cosine sweep: CUDA autodiff check,
# then every method at 0.03 s (1323 output samples, the CeL minimum is 1024).
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
"$KC_PYTHON" test/figure_fdtd.py --cuda
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/native_rate_compare.py \
  --output "$KC_SCRATCH/native-rate-smoke-${SLURM_JOB_ID}.json" \
  --plot-prefix "$KC_SCRATCH/native-rate-smoke-${SLURM_JOB_ID}" \
  --batches 1,2 --methods cached3,fdtd,diaz_44100,diaz_internal \
  --modes 100,123 --duration .03 --timeout 600 --excitation raised_cosine --loss mae \
  --python "$KC_SCRATCH/jax-venv/bin/python"
