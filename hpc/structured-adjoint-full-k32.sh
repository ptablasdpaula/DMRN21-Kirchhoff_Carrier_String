#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
batch="$1"
timeout --signal=TERM 40m "$KC_PYTHON" test/structured_adjoint_full_k32.py --batch "$batch" --evidence "$KC_SCRATCH/structured-adjoint-gpu-34510044.json" --output "$KC_SCRATCH/structured-adjoint-full-k32-b${batch}-${SLURM_JOB_ID}.json"
