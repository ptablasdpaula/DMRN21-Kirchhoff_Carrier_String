#!/bin/bash
# Usage: bash hpc/sound-matching-note.sh <note_str>   (one NSynth note, both families; fits the 1 h gpushort limit)
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
OUT="$KC_SCRATCH/sound-matching-notes-v2"
"$KC_PYTHON" experiments/applications/sound_matching/fit.py \
  --manifest /data/scratch/acw794/data/nsynth/selected/manifest.json \
  --note "$1" --families steel,nylon --starts 128 --duration 3.0 --seed 0 \
  --output "$OUT"
