#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export PIXI_CACHE_DIR="$KC_SCRATCH/octave-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
# pixi exec can rebuild its shared environment; serialize these jobs to avoid
# simultaneous deletion/recreation of the same conda metadata.
flock "$KC_SCRATCH/octave-reference-environment.lock" pixi exec --spec octave=10.3.0 -- "$KC_PYTHON" test/reference_paper_exact.py --mode "$1" --output "$KC_SCRATCH/paper-exact-$1-${SLURM_JOB_ID}.json"
