#!/bin/bash
set -euo pipefail
source hpc/environment.sh
# Keep this reproducible reference environment separate from older shared caches.
export PIXI_CACHE_DIR="$KC_SCRATCH/octave-cache"
pixi exec --spec octave=10.3.0 -- "$KC_PYTHON" test/kc_string.py
