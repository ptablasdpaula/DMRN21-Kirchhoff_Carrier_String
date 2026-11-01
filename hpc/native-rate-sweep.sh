#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/native_rate_compare.py \
  --output "$KC_SCRATCH/native-rate-sweep-${SLURM_JOB_ID}.json" \
  --plot-prefix "$KC_SCRATCH/native-rate-sweep-${SLURM_JOB_ID}" \
  --batches 1,2,4,8,16,32,64,128 --methods cached3,fdtd,diaz_44100 \
  --modes 100 --duration 1 --timeout 3600 --excitation raised_cosine --loss mae \
  --python "$KC_SCRATCH/jax-venv/bin/python"
