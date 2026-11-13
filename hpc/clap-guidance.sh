#!/bin/bash
# Usage: bash hpc/clap-guidance.sh <pair>   (pair keys in experiments/applications/semantic/prompts.py)
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
export HF_HOME=/data/scratch/acw794/home-cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$KC_PYTHON" experiments/applications/semantic/optimise.py --pair "$1" --grid production --starts 32 --seed 0 \
  --output "$KC_SCRATCH/clap/${1}-${SLURM_JOB_ID}"
