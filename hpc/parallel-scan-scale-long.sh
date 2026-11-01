#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" benchmarks/parallel_scan_scale.py --steps 16384 --output "$KC_SCRATCH/parallel-scan-scale-long-$SLURM_JOB_ID.json"
