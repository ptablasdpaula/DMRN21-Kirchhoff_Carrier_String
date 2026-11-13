#!/bin/bash
# Usage: bash hpc/native-rate-checkpointed.sh [methods] [batches] [tag] [duration]
# Checkpointed baselines for Fig. 1: the same one-second workload as
# hpc/native-rate-sweep.sh (raised cosine, MAE, ten parameters, 176 intervals at
# 88.2 kHz, 100 modes at 44.1 kHz) for the rematerialised variants fdtd_ckpt
# (torch.utils.checkpoint, 256-step chunks) and diaz_44100_ckpt (jax.checkpoint,
# 256-step segments). Batches can be split across tasks (one JSON each) and the
# JSONs merged with paper_dmrn2026/figures/merge_sweeps.py. No live plot: the
# per-case plot only knows the three original methods.
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
METHODS=${1:-fdtd_ckpt,diaz_44100_ckpt}
BATCHES=${2:-1,2,4,8,16,32,64,128}
TAG=${3:-all}
DURATION=${4:-1}
mkdir -p "$KC_SCRATCH/native-rate-ckpt"
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/native_rate_compare.py \
  --output "$KC_SCRATCH/native-rate-ckpt/${TAG}-${SLURM_JOB_ID}.json" \
  --batches "$BATCHES" --methods "$METHODS" \
  --modes 100 --duration "$DURATION" --timeout 3600 --excitation raised_cosine --loss mae \
  --python "$KC_SCRATCH/jax-venv/bin/python"
