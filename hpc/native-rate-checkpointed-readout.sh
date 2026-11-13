#!/bin/bash
# jaxdiffmodal checkpointed with the pickup read inside the scan (diaz_44100_ckptro):
# the JAX test first, then the one-second sweep over all batches.
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}" JAX_ENABLE_X64=true JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
"$KC_SCRATCH/jax-venv/bin/python" test/native_rate_study.py
bash hpc/native-rate-checkpointed.sh diaz_44100_ckptro 1,2,4,8,16,32,64,128 diaz-readout 1
