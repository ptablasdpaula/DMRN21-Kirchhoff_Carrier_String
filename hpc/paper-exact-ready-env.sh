#!/bin/bash
# Use the already-created Pixi Octave environment without asking Pixi to
# rebuild it while a long reference job is running.
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
# Octave's conda activation sets this to the environment prefix so that its
# standard-library .m files resolve when invoked outside `pixi exec`.
export OCTAVE_HOME="$(dirname "$(dirname "${environments[0]}")")"
"$KC_PYTHON" test/reference_paper_exact.py --mode gradients --output "$KC_SCRATCH/paper-exact-independent-gradients-${SLURM_JOB_ID}.json"
