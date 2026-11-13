#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" benchmarks/parallel_scan_scale.py --output "$KC_SCRATCH/parallel-scan-scale-$SLURM_JOB_ID.json"
