#!/bin/bash
# The supplement's runs, one per job:
#   sbatch hpc/job.sh hpc/applications.sh sound <NSynth note>     (both string families)
#   sbatch hpc/job.sh hpc/applications.sh semantic <prompt pair>  (pairs in applications/semantic/prompts.py)
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
OUT="$KC_SCRATCH/supplement"
case "$1" in
  sound)
    python -m applications.sound_matching.fit --manifest /data/scratch/acw794/data/nsynth/selected/manifest.json \
      --note "$2" --families steel,nylon --starts 128 --duration 3.0 --seed 0 --output "$OUT/sound_matching" ;;
  semantic)
    export HF_HOME=/data/scratch/acw794/home-cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python -m applications.semantic.optimise --pair "$2" --grid production --starts 32 --seed 0 \
      --output "$OUT/semantic/$2" ;;
  *) echo "usage: $0 sound <note> | semantic <pair>" >&2; exit 2 ;;
esac
