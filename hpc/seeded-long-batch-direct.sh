#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
"$KC_PYTHON" test/seeded_long_batch_direct.py --output "$KC_SCRATCH/seeded-long-batch-direct-${SLURM_JOB_ID}.json"
