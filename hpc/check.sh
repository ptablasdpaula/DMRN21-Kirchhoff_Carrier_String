#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" test/accelerated.py --device cuda --output "$KC_SCRATCH/validation-${SLURM_JOB_ID}.json"
"$KC_PYTHON" test/accelerated.py --device cpu --output "$KC_SCRATCH/validation-cpu-${SLURM_JOB_ID}.json"
for check in parameters_grid material_coordinates synthesis optimization numerical_validation; do
    "$KC_PYTHON" "test/$check.py"
done
"$KC_PYTHON" test/sound_matching_gpu.py --output "$KC_SCRATCH/sound-matching-check-${SLURM_JOB_ID}.json"
"$KC_PYTHON" test/clap_gpu_smoke.py --output "$KC_SCRATCH/clap-check-${SLURM_JOB_ID}.json"
"$KC_PYTHON" benchmarks/gpu.py --quick --repeats 3 --output "$KC_SCRATCH/quick-benchmark-${SLURM_JOB_ID}.json"
