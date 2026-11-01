#!/bin/bash
set -euo pipefail
source hpc/environment.sh
if [ ! -x "$KC_PYTHON" ]; then uv venv --python 3.12 "$KC_SCRATCH/venv"; fi
uv pip install --python "$KC_PYTHON" -r hpc/requirements.lock
"$KC_PYTHON" - <<'PY'
import torch,json
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name()),indent=2))
assert torch.cuda.is_available()
PY
uv pip freeze --python "$KC_PYTHON" > "$KC_SCRATCH/environment.freeze.txt"
