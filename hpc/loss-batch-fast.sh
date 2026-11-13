#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" benchmarks/loss_batch_grouped.py --output "$KC_SCRATCH/loss-batch-fast-${SLURM_JOB_ID}.json" --batches 128 --durations 4 --directions 2,6 --groups 8,16 --repeats 3
