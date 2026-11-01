#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
short="$KC_SCRATCH/even-batch-validation-${SLURM_JOB_ID}-short.json"
long="$KC_SCRATCH/even-batch-validation-${SLURM_JOB_ID}-long.json"
"$KC_PYTHON" test/exact_benchmark.py --device cuda --method cpu-modal,cuda-modal,cuda-seeded,cuda-register --output "$short"
"$KC_PYTHON" test/exact_benchmark.py --device cuda --method cpu-modal,cuda-modal,cuda-seeded,cuda-register --duration 4 --batch 1 --output "$long"
"$KC_PYTHON" benchmarks/exact_methods.py --methods cuda-modal,cuda-seeded,cuda-register --batches 1,64,128 --durations .25 --kinds synthesis,loss_backward --repeats 2 --validation "$long" --validation-short "$short" --output "$KC_SCRATCH/even-batch-quick-${SLURM_JOB_ID}.json"
