#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$KC_PYTHON" test/grouped_loss_contract.py
"$KC_PYTHON" test/manual_ce_loss.py --device cuda --output "$KC_SCRATCH/manual-ce-validation-${SLURM_JOB_ID}.json"
"$KC_PYTHON" benchmarks/manual_ce_compare.py --output "$KC_SCRATCH/manual-ce-compare-${SLURM_JOB_ID}.json" --batch 128 --duration 4 --groups 4,8,16 --repeats 3
