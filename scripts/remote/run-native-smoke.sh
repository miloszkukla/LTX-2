#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M0/native-smoke.json}
smoke_temp=$(mktemp -d /tmp/ltx-native-smoke.XXXXXX)
cleanup() {
    if [[ $smoke_temp == /tmp/ltx-native-smoke.* && -d $smoke_temp ]]; then
        rm -rf -- "$smoke_temp"
    fi
}
trap cleanup EXIT

compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
sass_architecture="sm_${compute_capability}"
native_source="$repo_root/scripts/remote/smoke/native/ltx_cuda_smoke.cu"
native_library="$smoke_temp/libltx_cuda_smoke.so"

nvcc \
    --shared \
    --std=c++17 \
    -Xcompiler=-fPIC \
    -gencode "arch=compute_${compute_capability},code=${sass_architecture}" \
    "$native_source" \
    -o "$native_library"

sass_listing=$(cuobjdump --list-elf "$native_library")
if ! grep -q "$sass_architecture" <<<"$sass_listing"; then
    echo "compiled library does not contain ${sass_architecture} SASS" >&2
    exit 1
fi
if grep -q 'ptx' <<<"$sass_listing"; then
    echo "compiled library unexpectedly contains PTX" >&2
    exit 1
fi
cuobjdump --dump-sass "$native_library" >/dev/null

project="$repo_root/scripts/remote/smoke/CudaSmoke/CudaSmoke.csproj"
dotnet restore "$project" --nologo
smoke_output=$(dotnet run --project "$project" --configuration Release --no-restore -- "$native_library")
if [[ $smoke_output != *"TorchSharp=pass PInvoke=pass CUDA=pass SASS=${sass_architecture}"* ]]; then
    echo "managed/native smoke did not report the complete pass line" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" \
SASS_ARCHITECTURE="$sass_architecture" \
NATIVE_LIBRARY="$native_library" \
NATIVE_SOURCE="$native_source" \
PROJECT="$project" \
SMOKE_OUTPUT="$smoke_output" \
python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "tests": {
        "torchsharp_tensor": "pass",
        "ldconfig_pinvoke": "pass",
        "cuda_kernel": "pass",
        "sass_only": "pass",
    },
    "torchsharp": {"package": "TorchSharp-cpu", "version": "0.107.0", "cuda_runtime_mixed": False},
    "native": {
        "target": os.environ["SASS_ARCHITECTURE"],
        "ptx_embedded": False,
        "library_sha256": sha256(os.environ["NATIVE_LIBRARY"]),
        "source_sha256": sha256(os.environ["NATIVE_SOURCE"]),
        "project_sha256": sha256(os.environ["PROJECT"]),
    },
    "output": os.environ["SMOKE_OUTPUT"],
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).splitlines()[-1],
    "dotnet_sdk": subprocess.check_output(["dotnet", "--version"], text=True).strip(),
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$smoke_output"
