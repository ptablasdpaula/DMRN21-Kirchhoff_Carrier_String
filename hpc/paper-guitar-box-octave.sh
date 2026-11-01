#!/bin/bash
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
shopt -s nullglob
environments=("${KC_PYTHON}"-*/bin/octave-cli)
if (( ${#environments[@]} != 1 )); then
    printf 'Expected exactly one ready Pixi Octave environment; found %s\n' "${#environments[@]}" >&2
    exit 1
fi
export PATH="$(dirname "${environments[0]}"):$PATH"
export OCTAVE_HOME="$(dirname "$(dirname "${environments[0]}")")"
"$KC_PYTHON" test/paper_guitar_box_octave.py --output "$KC_SCRATCH/paper-guitar-box-octave-${SLURM_JOB_ID}.json"
