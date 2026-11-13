#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
"$KC_PYTHON" test/sound_matching_gpu.py --output "$KC_SCRATCH/sound-matching-smoke-${SLURM_JOB_ID}.json"
"$KC_PYTHON" experiments/applications/sound_matching/fit.py \
  --manifest /data/scratch/acw794/data/nsynth/selected/manifest.json \
  --note guitar_acoustic_010-050-100 --families steel --smoke \
  --output "$KC_SCRATCH/sound-matching-smoke-${SLURM_JOB_ID}"
