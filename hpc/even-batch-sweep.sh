#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
duration=${1:?duration required}
start=${2:?first even batch required}
end=${3:?last even batch required}
[[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$start" -ge 2 && "$end" -le 128 && "$start" -le "$end" && $((start%2)) -eq 0 && $((end%2)) -eq 0 ]] || { echo 'invalid even batch range' >&2; exit 2; }
[[ "$duration" = .25 || "$duration" = 4 ]] || { echo 'invalid duration' >&2; exit 2; }
short="$KC_SCRATCH/even-sweep-${SLURM_JOB_ID}-short.json"
long="$KC_SCRATCH/even-sweep-${SLURM_JOB_ID}-long.json"
"$KC_PYTHON" test/exact_benchmark.py --device cuda --adjoint --method cpu-modal,cuda-modal,cuda-seeded,cuda-recompute,cuda-adjoint,cuda-register --output "$short"
"$KC_PYTHON" test/exact_benchmark.py --device cuda --adjoint --method cpu-modal,cuda-modal,cuda-seeded,cuda-recompute,cuda-adjoint,cuda-register --duration 4 --batch 1 --output "$long"
"$KC_PYTHON" test/even_batch_isolation.py
"$KC_PYTHON" benchmarks/even_batch_driver.py --start "$start" --end "$end" --duration "$duration" --repeats 3 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/even-sweep-${SLURM_JOB_ID}-${duration}-${start}-${end}.json"
