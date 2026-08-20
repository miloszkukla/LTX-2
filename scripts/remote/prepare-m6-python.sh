#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"$repo_root/scripts/remote/prepare-m4-python.sh" >/dev/null

"$repo_root/.venv/bin/python" - <<'PY'
import torch

if torch.__version__ != "2.13.0+cu132" or torch.version.cuda != "13.2":
    raise SystemExit(f"M6 oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"M6 requires exactly one CUDA GPU, found {torch.cuda.device_count()}")
PY

echo "M6 Python oracle prepared"
