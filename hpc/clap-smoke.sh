#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4 MPLBACKEND=Agg
export HF_HOME=/data/scratch/acw794/home-cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$KC_PYTHON" test/clap_frontend.py
"$KC_PYTHON" test/clap_gpu_smoke.py --output "$KC_SCRATCH/clap-smoke-${SLURM_JOB_ID}.json"
"$KC_PYTHON" experiments/applications/semantic/optimise.py --pair metallic_dull --grid pilot --starts 2 --max-iter 6 \
  --output "$KC_SCRATCH/clap-smoke-${SLURM_JOB_ID}"
