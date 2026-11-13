#!/bin/bash
set -euo pipefail
source hpc/environment.sh
device=$1
backend=$2
"$KC_PYTHON" experiments/recovery.py --device "$device" --backend "$backend" --output "$KC_SCRATCH/recovery-$SLURM_JOB_ID/$device-$backend"
