#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
sha256sum experiments/scalar_cache/*.py experiments/scalar_cache/*.cu test/scalar_cache_smoke.py > "$KC_SCRATCH/scalar-cache-source-${SLURM_JOB_ID}.sha256"
"$KC_PYTHON" test/scalar_cache_smoke.py
sha256sum -c "$KC_SCRATCH/scalar-cache-source-${SLURM_JOB_ID}.sha256"
