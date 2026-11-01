#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1
"$KC_PYTHON" test/reconstruction_coordinates.py --output "$KC_SCRATCH/reconstruction-coordinates-${SLURM_JOB_ID}.json"
