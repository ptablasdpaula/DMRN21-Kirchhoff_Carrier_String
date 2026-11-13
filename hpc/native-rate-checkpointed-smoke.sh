#!/bin/bash
# Smoke test before the checkpointed-baseline sweep: the two CPU/GPU tests of
# the new code paths, then a 50 ms run of both methods at batches 1 and 2
# (the worker validates fdtd_ckpt against the replay adjoint).
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
"$KC_PYTHON" test/figure_fdtd.py --cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}" JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
"$KC_SCRATCH/jax-venv/bin/python" test/native_rate_study.py
export MAX_JOBS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
bash hpc/native-rate-checkpointed.sh fdtd_ckpt,diaz_44100_ckpt 1,2 smoke 0.05
