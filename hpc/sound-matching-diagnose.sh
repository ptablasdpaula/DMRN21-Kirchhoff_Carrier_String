#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
RUN=/data/scratch/acw794/kc-string-gpu/sound-matching-34913443/guitar_acoustic_010-040-050
"$KC_PYTHON" experiments/applications/sound_matching/diagnose.py --fit "$RUN/steel/fit.json" --target "$RUN/target.wav" \
  --label-f0 82.4068892282175 --family steel --output "$KC_SCRATCH/diagnose-E2-steel-${SLURM_JOB_ID}.json"
