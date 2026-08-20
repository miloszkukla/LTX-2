#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_path="$repo_root/.venv/bin/python"

"$python_path" - <<'PY'
from pathlib import Path
import torch

site = Path(torch.__file__).resolve().parent.parent
paths = [Path(torch.__file__).resolve().parent / "lib"]
paths.extend(sorted(path for path in (site / "nvidia").glob("*/lib") if path.is_dir()))
print(":".join(str(path) for path in paths))
PY
