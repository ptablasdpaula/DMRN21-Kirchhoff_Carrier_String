#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
sha256sum experiments/scalar_cache/*.py experiments/scalar_cache/*.cu test/scalar_cache_fallback.py test/scalar_cache_coordinates.py > "$KC_SCRATCH/scalar-cache-extra-source-${SLURM_JOB_ID}.sha256"
"$KC_PYTHON" test/scalar_cache_fallback.py
"$KC_PYTHON" test/scalar_cache_coordinates.py "$KC_SCRATCH/scalar-cache-coordinates-${SLURM_JOB_ID}.json"
sha256sum -c "$KC_SCRATCH/scalar-cache-extra-source-${SLURM_JOB_ID}.sha256"
"$KC_PYTHON" benchmarks/scalar_fallback_compare.py "$KC_SCRATCH/scalar-fallback-compare-${SLURM_JOB_ID}.json"
