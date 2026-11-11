#!/bin/bash
# Usage: bash hpc/native-rate-jit-smoke.sh jax | jax-sine | torch-test | torch MODE BLOCK [reentrant]
# Smoke runs for the compiled autodiff baselines of hpc/native-rate-jit.sh: batch 1,
# a quarter of a second, to check correctness and to choose the settings of the
# sweep (the JAX scan's unroll; torch.compile's block length, mode and checkpoint
# form). A failing configuration is reported and does not stop the others.
set -uo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
export PYTHONPATH="$KC_SCRATCH/diaz-deps${PYTHONPATH:+:$PYTHONPATH}"
export TIMEOUT=${TIMEOUT:-1500}
PYTHON="$KC_SCRATCH/jax-venv/bin/python"
case "${1:?jax, torch-test or torch}" in
  jax)
    # test/figure_fdtd_jax.py (the twin against the PyTorch baseline) runs on CPU under srun; not repeated here.
    for unroll in 1 8 32; do
      bash hpc/native-rate-jit.sh fdtd_jax_ckpt,fdtd_jax 1 "smoke-jax-u${unroll}" 0.25 --unroll "$unroll" \
        || echo "FAILED: jax unroll $unroll"
    done ;;
  jax-sine)
    for unroll in 1 8 32 64; do
      bash hpc/native-rate-jit.sh fdtd_jax_sine_ckpt,fdtd_jax_sine 1 "smoke-jax-sine-u${unroll}" 0.25 --unroll "$unroll" \
        || echo "FAILED: jax sine unroll $unroll"
    done ;;
  torch-test)
    CACHE="${TMPDIR:-/tmp}/kc-jit-test-${SLURM_JOB_ID:-$$}"
    export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    "$PYTHON" test/figure_fdtd.py --cuda --compiled || echo "FAILED: torch.compile default mode"
    "$PYTHON" test/figure_fdtd.py --cuda --compiled-cudagraphs || echo "FAILED: torch.compile reduce-overhead mode" ;;
  torch)
    MODE=${2:?mode}; BLOCK=${3:?block}
    EXTRA=(--compile-block "$BLOCK" --compile-mode "$MODE"); NAME="smoke-torch-${MODE}-b${BLOCK}"
    if [ "${4:-}" = reentrant ]; then EXTRA+=(--reentrant); NAME="${NAME}-reentrant"; fi
    bash hpc/native-rate-jit.sh fdtd_ckpt_compiled 1 "$NAME" 0.25 "${EXTRA[@]}" || echo "FAILED: $NAME" ;;
  *) echo "unknown smoke $1"; exit 2 ;;
esac
