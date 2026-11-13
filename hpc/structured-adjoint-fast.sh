#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" test/structured_adjoint_gpu.py --output "$KC_SCRATCH/structured-adjoint-fast-${SLURM_JOB_ID}.json" --batches 1 --chunks 16,32,64 --window 4 --repeats 2
