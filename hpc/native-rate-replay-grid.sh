#!/bin/bash
# Usage: bash hpc/native-rate-replay-grid.sh [METHODS] [BATCHES] [TAG]
# The hand-written adjoint on the grid (src/adjoint_grid.py, kc_adjoint_grid.cu):
# the kernel of hpc/native-rate-replay.sh without mode stepping, on the same
# one-second workload. The worker validates every case against the modal kernel
# (replay) before timing it. With METHODS=replay_grid,replay the modal kernel is
# timed in the same job, for a same-node pair.
set -euo pipefail
bash hpc/native-rate-checkpointed.sh "${1:-replay_grid}" "${2:-1,2,4,8,16,32,64,128}" "${3:-replay-grid}" 1
