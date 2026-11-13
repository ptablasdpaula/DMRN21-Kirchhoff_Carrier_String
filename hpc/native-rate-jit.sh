#!/bin/bash
# Usage: bash hpc/native-rate-jit.sh [methods] [batches] [tag] [duration] [native_rate_compare.py options...]
# Compiled autodiff baselines on the one-second workload of hpc/native-rate-sweep.sh
# (raised cosine, MAE, ten parameters, 176 intervals at 88.2 kHz): the spatial
# FDTD scheme of the PyTorch baseline run through a compiler instead of eagerly.
#   fdtd_compiled, fdtd_ckpt_compiled   torch.compile blocks (--compile-block, --compile-mode, --reentrant)
#   fdtd_jax, fdtd_jax_ckpt             the same scheme as a jax.jit scan (--unroll)
#   fdtd_jax_sine, fdtd_jax_sine_ckpt   the same scan in the discrete sine basis
#   fdtd_jax_eager                      fdtd_jax as a Python loop without jax.jit
#   fdtd_ckpt_reentrant                 eager fdtd_ckpt with the reentrant form of torch.utils.checkpoint
# The _ckpt variants rematerialise 256-step segments, as fdtd_ckpt and the CUDA
# adjoint do. Batches can be split across tasks (one JSON each) and merged with
# paper_dmrn2026/figures/merge_sweeps.py. TIMEOUT sets the per-case limit in seconds.
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
# Compiler caches on node-local disk and fresh for every job, so that no
# compiled artefact is reused from an earlier run.
CACHE="${TMPDIR:-/tmp}/kc-jit-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$KC_SCRATCH/native-rate-jit"
METHODS=${1:-fdtd_jax_ckpt,fdtd_ckpt_compiled}
BATCHES=${2:-1,2,4,8,16,32,64,128}
TAG=${3:-jit}
DURATION=${4:-1}
"$KC_SCRATCH/jax-venv/bin/python" benchmarks/native_rate_compare.py \
  --output "$KC_SCRATCH/native-rate-jit/${TAG}-${SLURM_JOB_ID:-local}.json" \
  --batches "$BATCHES" --methods "$METHODS" \
  --modes 100 --duration "$DURATION" --timeout "${TIMEOUT:-3600}" --excitation raised_cosine --loss mae \
  --python "$KC_SCRATCH/jax-venv/bin/python" "${@:5}"
