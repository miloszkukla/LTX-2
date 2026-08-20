#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

scripts/remote/prepare-m1-abi.sh
python_path="$repo_root/.venv/bin/python"
if ! "$python_path" - <<'PY' >/dev/null 2>&1
import numpy
import packaging
import safetensors

raise SystemExit(
    0
    if safetensors.__version__ == "0.6.2"
    and numpy.__version__ == "2.3.2"
    and packaging.__version__ == "25.0"
    else 1
)
PY
then
    uv pip install --python "$python_path" \
        safetensors==0.6.2 \
        numpy==2.3.2 \
        packaging==25.0
fi

"$python_path" - <<'PY'
import numpy
import packaging
import safetensors
import torch

versions = {
    "numpy": numpy.__version__,
    "packaging": packaging.__version__,
    "safetensors": safetensors.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
expected = {
    "numpy": "2.3.2",
    "packaging": "25.0",
    "safetensors": "0.6.2",
    "torch": "2.13.0+cu132",
    "cuda": "13.2",
}
if versions != expected:
    raise SystemExit(f"M2 Python dependency mismatch: {versions}")
PY

echo "M2 Python storage oracle prepared"
