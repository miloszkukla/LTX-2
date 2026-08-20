#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"$repo_root/scripts/remote/prepare-m3-python.sh"

"$repo_root/.venv/bin/python" - <<'PY'
import torch

if torch.__version__ != "2.13.0+cu132" or torch.version.cuda != "13.2":
    raise SystemExit(f"M4 oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("M4 CUDA oracle is unavailable")
if not hasattr(torch, "float8_e4m3fn"):
    raise SystemExit("M4 oracle lacks float8_e4m3fn")
PY

echo "M4 Python oracle prepared"
