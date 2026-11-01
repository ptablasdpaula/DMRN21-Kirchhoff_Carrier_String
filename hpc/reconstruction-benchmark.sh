#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
case "${1:-}" in
  smoke)
    batches='1,8,64,128'; durations='.25,4'; checkpoints='256'; repeats=3 ;;
  checkpoints)
    batches='1,128'; durations='4'; checkpoints='32,64,128,256,512,1024'; repeats=3 ;;
  *) echo 'Expected smoke or checkpoints.' >&2; exit 2 ;;
esac
validation="$KC_SCRATCH/reconstruction-validation-${SLURM_JOB_ID}.json"
output="$KC_SCRATCH/reconstruction-benchmark-${1}-${SLURM_JOB_ID}.json"
"$KC_PYTHON" test/reconstruction.py --output "$validation"
"$KC_PYTHON" benchmarks/reconstruction.py --validation "$validation" --batches "$batches" --durations "$durations" --checkpoints "$checkpoints" --repeats "$repeats" --output "$output"
