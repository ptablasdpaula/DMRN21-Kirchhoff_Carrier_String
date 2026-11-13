#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" experiments/parallel_segments/test_toy_scan.py cpu
"$KC_PYTHON" experiments/parallel_segments/test_toy_scan.py cuda
"$KC_PYTHON" experiments/parallel_segments/bench_toy_scan.py --device cuda --repeats 3 --output "$KC_SCRATCH/parallel-toy-$SLURM_JOB_ID.json"
