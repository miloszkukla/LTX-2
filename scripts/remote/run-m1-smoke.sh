#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M1/abi-smoke.json}
cd "$repo_root"

scripts/remote/prepare-m1-abi.sh
torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
cuda_library="$repo_root/build/M1/ltx-cuda/libltx_cuda.so"
compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M1/ltx-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

sass_listing=$(cuobjdump --list-elf "$cuda_library")
if ! grep -q "sm_${compute_capability}" <<<"$sass_listing" || grep -q 'ptx' <<<"$sass_listing"; then
    echo "libltx_cuda must contain only the worker SASS target" >&2
    exit 1
fi

smoke_output=$(dotnet run \
    --project tests/Ltx.AbiSmoke/Ltx.AbiSmoke.csproj \
    --configuration Release \
    --no-build \
    -- "$torchsharp_library" "$cuda_library")
if [[ $smoke_output != *"PyTorch=2.13.0 CUDA=13.2 GPU=sm_${compute_capability} native=pass"* ]]; then
    echo "M1 ABI smoke did not report the complete pass line" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" \
TORCHSHARP_LIBRARY="$torchsharp_library" \
CUDA_LIBRARY="$cuda_library" \
COMPUTE_CAPABILITY="$compute_capability" \
SMOKE_OUTPUT="$smoke_output" \
"$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "abi": {
        "ltx": "1.0",
        "torchsharp_managed": "0.107.0",
        "torchsharp_revision": "8f4def03b641b6753f18076aa5438f8eaaef2d30",
        "pytorch": "2.13.0",
        "cuda": "13.2",
        "cxx11_abi": 1,
    },
    "gpu": {"sass_architecture": f"sm_{os.environ['COMPUTE_CAPABILITY']}"},
    "tests": {
        "torchsharp_cuda_tensor": "pass",
        "torchsharp_abi_probe": "pass",
        "torchsharp_native_add_linear": "pass",
        "ltx_native_cuda": "pass",
        "sass_only": "pass",
    },
    "libraries": {
        "libLibTorchSharp_sha256": sha256(os.environ["TORCHSHARP_LIBRARY"]),
        "libltx_cuda_sha256": sha256(os.environ["CUDA_LIBRARY"]),
    },
    "output": os.environ["SMOKE_OUTPUT"],
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$smoke_output"
