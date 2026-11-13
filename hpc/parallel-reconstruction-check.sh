#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
"$KC_PYTHON" test/parallel_reconstruction.py --output "$KC_SCRATCH/parallel-reconstruction-validation-${SLURM_JOB_ID}.json"
