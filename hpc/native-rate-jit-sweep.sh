#!/bin/bash
# Usage: bash hpc/native-rate-jit-sweep.sh METHOD [BATCHES]
# The compiled-autodiff rows of the paper figure: one method of
# hpc/native-rate-jit.sh over the batch sweep on the one-second workload, with
# the settings chosen from the smoke runs (hpc/native-rate-jit-smoke.sh). One
# JSON per call; merge them with paper_dmrn2026/figures/merge_sweeps.py.
set -euo pipefail
UNROLL_GRID=32      # jax.lax.scan unroll of fdtd_jax, fdtd_jax_ckpt (smoke: 1.67, 0.87, 0.77 s at 1, 8, 32)
UNROLL_SINE=64      # jax.lax.scan unroll of fdtd_jax_sine, fdtd_jax_sine_ckpt (smoke: 1.47, 0.61, 0.54, 0.51 s at 1, 8, 32, 64)
BLOCK=${BLOCK:-16}  # torch.compile block length of the *_compiled methods (smoke: 256 steps do not compile in 25 min)
MODE=default        # torch.compile mode (smoke: reduce-overhead's CUDA graphs fail under checkpointing)
REENTRANT=(--reentrant)  # reentrant torch.utils.checkpoint (smoke: 4.93 against 8.10 s; job 35744004 ran the non-reentrant form)
METHOD=${1:?method}
BATCHES=${2:-1,2,4,8,16,32,64,128}
TAG="${TAG:-${METHOD}-b${BATCHES//,/_}}"
case "$METHOD" in
  fdtd_jax|fdtd_jax_ckpt)
    bash hpc/native-rate-jit.sh "$METHOD" "$BATCHES" "$TAG" 1 --unroll "$UNROLL_GRID" ;;
  fdtd_jax_sine|fdtd_jax_sine_ckpt)
    bash hpc/native-rate-jit.sh "$METHOD" "$BATCHES" "$TAG" 1 --unroll "$UNROLL_SINE" ;;
  fdtd_ckpt_reentrant)
    bash hpc/native-rate-jit.sh "$METHOD" "$BATCHES" "$TAG" 1 ;;
  fdtd_compiled|fdtd_ckpt_compiled|fdtd_sine_compiled|fdtd_sine_ckpt_compiled)
    bash hpc/native-rate-jit.sh "$METHOD" "$BATCHES" "$TAG" 1 --compile-block "$BLOCK" --compile-mode "$MODE" "${REENTRANT[@]}" ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
