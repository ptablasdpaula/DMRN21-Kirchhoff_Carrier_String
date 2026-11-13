#!/bin/bash
# Accuracy/convergence studies on a compute node (CPU): the native kernel is
# built there; the modal reference runs jaxdiffmodal on the CPU JAX backend.
# Submit with
#   sbatch --partition=compute --cpus-per-task=4 --mem=32G --time=02:30:00 \
#          --job-name=kc-accuracy --output=$KC_SCRATCH/kc-accuracy-%j.log \
#          --wrap="bash hpc/accuracy-study.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.pixi/envs/default/bin:$PATH"   # ninja
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=/data/scratch/acw794/kc-string-gpu/extensions/cpu-tests
export PYTHONPATH=/data/scratch/acw794/kc-string-gpu/diaz-deps JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
mkdir -p "$TORCH_EXTENSIONS_DIR"
PY=/data/scratch/acw794/kc-string-gpu/jax-venv/bin/python
status=0
run() { echo; echo "=== $* ($(date -u +%H:%M:%S))"; "$@" || { echo "FAILED: $*"; status=1; }; }
run "$PY" benchmarks/accuracy_study.py --parameters nominal --order 4
run "$PY" benchmarks/accuracy_study.py --parameters batch0 --order 4
run "$PY" benchmarks/accuracy_study.py --parameters nominal --order 2
run "$PY" benchmarks/stencil_study.py --cases 176:2:2,176:2:4,176:3:2,176:3:4,176:6:4 \
    --output benchmarks/accuracy_study_results/stencil-1s-nominal-api.json
echo; echo "accuracy studies exit $status ($(date -u +%H:%M:%S))"
exit $status
