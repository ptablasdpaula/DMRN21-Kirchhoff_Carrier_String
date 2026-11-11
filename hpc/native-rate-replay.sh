#!/bin/bash
# Replay adjoint (src/adjoint.py: the CUDA kernel with 256-step checkpoints and
# no per-step scalar cache) as a timed method on the one-second workload of
# hpc/native-rate-sweep.sh: the autodiff-against-replay GPU test first, then the
# sweep over all batches. The worker's own validation is replay against itself;
# the test and the existing fdtd rows tie replay to autodiff.
set -euo pipefail
source hpc/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MAX_JOBS=4
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
"$KC_PYTHON" test/figure_fdtd.py --cuda
bash hpc/native-rate-checkpointed.sh replay 1,2,4,8,16,32,64,128 replay 1
