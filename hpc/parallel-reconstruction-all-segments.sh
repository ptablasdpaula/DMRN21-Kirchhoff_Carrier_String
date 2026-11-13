#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
validation="$KC_SCRATCH/parallel-reconstruction-validation-${SLURM_JOB_ID}.json"
output="$KC_SCRATCH/parallel-reconstruction-all-segments-${SLURM_JOB_ID}.json"
"$KC_PYTHON" test/parallel_reconstruction.py --output "$validation"
"$KC_PYTHON" benchmarks/parallel_reconstruction.py --validation "$validation" --batches 1 --durations 4 --checkpoints 10 --groups 65535 --repeats 3 --output "$output"
