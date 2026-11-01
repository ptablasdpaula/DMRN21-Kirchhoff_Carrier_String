#!/bin/bash
# Source inside an allocated GPU job. All environment/build artifacts live on scratch.
set -euo pipefail
module load cuda/13.1.1-none-none
export KC_SCRATCH=/data/scratch/acw794/kc-string-gpu
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCH_EXTENSIONS_DIR="$KC_SCRATCH/extensions/$(hostname)/torch-2.14.1-cu131"
mkdir -p "$KC_SCRATCH" "$TORCH_EXTENSIONS_DIR"
export KC_PYTHON="$KC_SCRATCH/venv/bin/python"
export PATH="$KC_SCRATCH/venv/bin:$PATH"
