#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/alpha_scan_compare.py --validation "$KC_SCRATCH/alpha-scan-check-34508178.json" --output "$KC_SCRATCH/alpha-scan-fast-${SLURM_JOB_ID}.json" --durations .25,4 --batches 128 --checkpoint 256 --groups 8 --loss-group 16 --repeats 3 --max-estimated-seconds 120
