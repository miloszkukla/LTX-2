#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

scripts/remote/prepare-m2-python.sh
"$repo_root/.venv/bin/python" - <<'PY'
import safetensors
import torch

versions = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "safetensors": safetensors.__version__,
}
expected = {
    "torch": "2.13.0+cu132",
    "cuda": "13.2",
    "safetensors": "0.6.2",
}
if versions != expected or not torch.cuda.is_available():
    raise SystemExit(f"M3 Python oracle mismatch: {versions}, cuda_available={torch.cuda.is_available()}")
PY

echo "M3 Python core-model oracle prepared"
