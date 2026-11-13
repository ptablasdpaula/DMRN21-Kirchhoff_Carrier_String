#!/bin/bash
set -euo pipefail
source hpc/environment.sh
"$KC_PYTHON" test/accelerated.py --device cuda --output "$KC_SCRATCH/validation-notebooks-${SLURM_JOB_ID}.json"
"$KC_PYTHON" benchmarks/gpu.py --devices cuda --output "$KC_SCRATCH/benchmark-notebooks-${SLURM_JOB_ID}.json" --repeats 5
"$KC_PYTHON" -m ipykernel install --prefix "$KC_SCRATCH/venv" --name python3
export JUPYTER_PATH="$KC_SCRATCH/venv/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
"$KC_PYTHON" test/optimization_workbench.py
"$KC_PYTHON" test/kc_string_forward.py
bash hpc/octave.sh
