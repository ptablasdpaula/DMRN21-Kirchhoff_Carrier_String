#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
"$KC_SCRATCH/jax-venv/bin/python" test/jax_loss_reference.py --device cuda --output "$KC_SCRATCH/jax-loss-reference-${SLURM_JOB_ID}.json"
