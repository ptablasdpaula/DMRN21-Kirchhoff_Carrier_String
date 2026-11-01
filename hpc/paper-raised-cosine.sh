#!/bin/bash
# Octave reference comparison for the raised-cosine pluck.
# $1: long | gradients; $2: spatial order (2 = Russo et al.'s three-point code at
# 132.3 kHz, 4 = the five-point operator substituted into the same code at 88.2 kHz);
# $3: optional oversampling override (the gradient test's perturbed string sits just
# outside the 2x margin with the five-point stencil, so it runs at 3x).
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=1
ORDER="${2:-2}"
OVERSAMPLING_ARGS=()
if [[ -n "${3:-}" ]]; then OVERSAMPLING_ARGS=(--oversampling "$3"); fi
shopt -s nullglob
environments=("${KC_PYTHON}"-*/bin/octave-cli)
if (( ${#environments[@]} != 1 )); then
    printf 'Expected exactly one ready Pixi Octave environment; found %s\n' "${#environments[@]}" >&2
    exit 1
fi
export PATH="$(dirname "${environments[0]}"):$PATH"
export OCTAVE_HOME="$(dirname "$(dirname "${environments[0]}")")"
"$KC_PYTHON" test/reference_paper_exact.py --mode "$1" --excitation raised_cosine --order "$ORDER" "${OVERSAMPLING_ARGS[@]}" \
    --output "$KC_SCRATCH/paper-raised-$1-order$ORDER${3:+-x$3}-${SLURM_JOB_ID}.json"
