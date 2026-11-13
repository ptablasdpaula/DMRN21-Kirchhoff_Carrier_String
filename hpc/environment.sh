#!/bin/bash
# Source inside an allocated job: CUDA, the scratch environment and the kernel build cache.
set -euo pipefail
module load cuda/13.1.1-none-none 2>/dev/null || true
export KC_SCRATCH=/data/scratch/acw794/kc-string-gpu
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCH_EXTENSIONS_DIR="$KC_SCRATCH/extensions/$(hostname)/torch-2.14.1-cu131"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$KC_SCRATCH" "$TORCH_EXTENSIONS_DIR"
export PATH="$KC_SCRATCH/env/bin:$PATH"
