#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
setting=${1:?expected smoke or a checkpoint length}
case "$setting" in
  smoke)
    batches='1,8'; durations='.25'; checkpoints='10,256'; groups='16'; repeats=3 ;;
  10|64|256)
    batches='1,128'; durations='4'; checkpoints="$setting"; groups='1,16,64'; repeats=3 ;;
  *) echo 'Expected smoke, 10, 64 or 256.' >&2; exit 2 ;;
esac
validation="$KC_SCRATCH/parallel-reconstruction-validation-${SLURM_JOB_ID}.json"
output="$KC_SCRATCH/parallel-reconstruction-benchmark-${setting}-${SLURM_JOB_ID}.json"
"$KC_PYTHON" test/parallel_reconstruction.py --output "$validation"
"$KC_PYTHON" benchmarks/parallel_reconstruction.py --validation "$validation" --batches "$batches" --durations "$durations" --checkpoints "$checkpoints" --groups "$groups" --repeats "$repeats" --output "$output"
