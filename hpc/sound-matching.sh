#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
OUT="$KC_SCRATCH/sound-matching-${SLURM_JOB_ID}"
"$KC_PYTHON" experiments/applications/sound_matching/fit.py \
  --manifest /data/scratch/acw794/data/nsynth/selected/manifest.json \
  --families steel,nylon --starts 128 --duration 3.0 --seed 0 \
  --output "$OUT"
