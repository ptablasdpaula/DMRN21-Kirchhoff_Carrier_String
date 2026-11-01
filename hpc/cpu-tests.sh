#!/bin/bash
# CPU test battery for a compute node: the login node kills the C++ extension
# builds, so everything that compiles a kernel runs here. Submit with
#   sbatch --partition=compute --cpus-per-task=4 --mem=32G --time=03:00:00 \
#          --job-name=kc-cpu-tests --output=$KC_SCRATCH/kc-cpu-tests-%j.log \
#          --wrap="bash hpc/cpu-tests.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.pixi/envs/default/bin:$PATH"   # ninja
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=/data/scratch/acw794/kc-string-gpu/extensions/cpu-tests
mkdir -p "$TORCH_EXTENSIONS_DIR"
PY=/data/scratch/acw794/kc-string-gpu/venv/bin/python
JAXPY=/data/scratch/acw794/kc-string-gpu/jax-venv/bin/python
status=0
run() { echo; echo "=== $* ($(date -u +%H:%M:%S))"; "$@" || { echo "FAILED: $*"; status=1; }; }
run "$PY" test/positions.py
run "$PY" test/grouped_loss_contract.py
run "$PY" test/kc_string_forward.py
run "$PY" test/numerical_validation.py
run "$PY" test/parameters_grid.py
run "$PY" test/synthesis.py
run "$PY" test/figure_fdtd.py
run "$PY" test/accelerated.py --device cpu
run "$PY" test/alpha_scan_math.py
run "$PY" test/seeded.py
run "$PY" test/material_coordinates.py
run "$PY" test/native_rate_plot.py
run "$PY" test/figure_comparison_plot.py
run "$PY" test/figure_comparison_driver.py
export PYTHONPATH=/data/scratch/acw794/kc-string-gpu/diaz-deps JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
run "$JAXPY" test/jax_loss_reference.py --device cpu --output "$TORCH_EXTENSIONS_DIR/jax-loss-reference-${SLURM_JOB_ID:-local}.json"
run "$JAXPY" test/accuracy_study.py
run "$JAXPY" test/native_rate_study.py
run "$JAXPY" test/figure_diaz.py
echo; echo "cpu tests exit $status ($(date -u +%H:%M:%S))"
exit $status
