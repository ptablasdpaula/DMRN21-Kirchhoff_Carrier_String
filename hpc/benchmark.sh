#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" test/accelerated.py --device cuda --output "$KC_SCRATCH/validation-${SLURM_JOB_ID}.json"
"$KC_PYTHON" benchmarks/gpu.py --output "$KC_SCRATCH/benchmark-${SLURM_JOB_ID}.json" --repeats 5
"$KC_PYTHON" benchmarks/profile_gpu.py --output "$KC_SCRATCH/profile-${SLURM_JOB_ID}"
# Revalidate the exact code used by the measurement process, also recording its hash.
"$KC_PYTHON" test/accelerated.py --device cuda --output "$KC_SCRATCH/validation-final-${SLURM_JOB_ID}.json"
