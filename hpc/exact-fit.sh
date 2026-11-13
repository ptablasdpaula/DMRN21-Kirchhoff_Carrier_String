#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
method=${1:?candidate required}
case "$method" in
  cuda-recompute|cuda-seeded) extra=() ;;
  cuda-adjoint) extra=(--adjoint) ;;
  *) echo "Invalid candidate: $method" >&2; exit 2 ;;
esac
short="$KC_SCRATCH/exact-fit-${method}-${SLURM_JOB_ID}-short.json"
long="$KC_SCRATCH/exact-fit-${method}-${SLURM_JOB_ID}-long.json"
"$KC_PYTHON" test/exact_benchmark.py --device cuda "${extra[@]}" --method "cuda-modal,$method" --output "$short"
"$KC_PYTHON" test/exact_benchmark.py --device cuda "${extra[@]}" --method "cuda-modal,$method" --duration 4 --batch 1 --output "$long"
"$KC_PYTHON" experiments/exact_recovery.py --candidate "$method" --output "$KC_SCRATCH/exact-fit-${method}-${SLURM_JOB_ID}"
