#!/bin/bash
# Build the scratch environment once (any node; it needs no GPU): bash hpc/setup.sh
set -euo pipefail
source hpc/environment.sh
[ -x "$KC_SCRATCH/env/bin/python" ] || uv venv --python 3.12 "$KC_SCRATCH/env"
uv pip install --python "$KC_SCRATCH/env/bin/python" -r hpc/requirements.txt -e .
uv pip freeze --python "$KC_SCRATCH/env/bin/python" > hpc/requirements.lock
python -c "import torch, jax; print(torch.__version__, torch.cuda.is_available(), jax.__version__)"
