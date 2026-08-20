#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M4/cuda-quantization-lora.json}
fixture="$repo_root/tests/fixtures/M4"
cd "$repo_root"

scripts/remote/prepare-m4-python.sh
fixture_temp=$(mktemp -d /tmp/ltx-m4-fixture.XXXXXX)
cleanup() {
    if [[ $fixture_temp == /tmp/ltx-m4-fixture.* && -d $fixture_temp ]]; then
        rm -rf -- "$fixture_temp"
    fi
}
trap cleanup EXIT

"$repo_root/.venv/bin/python" scripts/parity/generate-m4-fixtures.py "$fixture_temp"
tracked_files=$(find "$fixture" -maxdepth 1 -type f -printf '%f\n' | sort)
generated_files=$(find "$fixture_temp" -maxdepth 1 -type f -printf '%f\n' | sort)
if [[ $tracked_files != "$generated_files" ]]; then
    echo "tracked M4 fixture file set does not match the pinned Python oracle" >&2
    exit 1
fi
while IFS= read -r name; do
    if ! cmp -s "$fixture/$name" "$fixture_temp/$name"; then
        echo "tracked M4 fixture '$name' does not match the pinned Python oracle" >&2
        exit 1
    fi
done <<<"$tracked_files"

"$repo_root/.venv/bin/python" - <<'PY'
import json
from pathlib import Path

surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
manifest = json.loads(Path("tests/fixtures/M4/manifest.json").read_text())
in_scope = [item["name"] for item in surface["native_kernels"] if item["status"] == "in_scope"]
deferred = [item for item in surface["native_kernels"] if item["status"] == "deferred"]
bindings = [item["name"] for item in surface["native_bindings"] if item["status"] == "in_scope"]
if in_scope != manifest["native_kernel_inventory"] or len(in_scope) != 24:
    raise SystemExit("M4 fixture does not cover the exact M0 in-scope native-kernel inventory")
if bindings != manifest["native_binding_inventory"] or len(bindings) != 15:
    raise SystemExit("M4 fixture does not cover the exact M0 in-scope native-binding inventory")
if len(deferred) != 5 or any("B200" not in item["reason"] and "Multi-GPU" not in item["reason"] for item in deferred):
    raise SystemExit("M4 deferred-kernel set differs from the accepted M0 inventory")
PY

library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M1/ltx-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
m4_output=$(dotnet run \
    --project tests/Ltx.CudaQuantizationTests/Ltx.CudaQuantizationTests.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture")
if [[ $m4_output != *"M4 CUDA/quantization/LoRA fixtures: 45 passed, 0 failed; kernels=24, bindings=15, quantization=4, lora=2"* ]]; then
    echo "M4 C# runner did not report the complete forty-five-check pass" >&2
    exit 1
fi

cuda_library="$repo_root/build/M1/ltx-cuda/libltx_cuda.so"
compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
sass_listing=$(cuobjdump --list-elf "$cuda_library")
if ! grep -q "sm_${compute_capability}" <<<"$sass_listing" || grep -q 'ptx' <<<"$sass_listing"; then
    echo "M4 libltx_cuda must contain only the worker SASS target" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" M4_OUTPUT="$m4_output" COMPUTE_CAPABILITY="$compute_capability" \
"$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

fixture = Path(os.environ["FIXTURE"])
manifest = json.loads((fixture / "manifest.json").read_text())
surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(fixture.iterdir())
    if path.is_file()
}
digest = hashlib.sha256()
for name, checksum in sorted(checksums.items()):
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(checksum.encode())
    digest.update(b"\0")

in_scope = [item["name"] for item in surface["native_kernels"] if item["status"] == "in_scope"]
deferred = [item for item in surface["native_kernels"] if item["status"] == "deferred"]
bindings = [item["name"] for item in surface["native_bindings"] if item["status"] == "in_scope"]
kernel_correctness = {name: "gpu_fixture_pass" for name in in_scope}
binding_correctness = {name: "managed_binding_pass" for name in bindings}
report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "fixture_revision": manifest["fixture_revision"],
    "fixture_set_sha256": digest.hexdigest(),
    "fixture_sha256": checksums,
    "oracle": manifest["oracle"],
    "dependencies": {
        "torch": "2.13.0+cu132",
        "torchsharp": "0.107.0",
        "cuda": "13.2",
    },
    "gpu": {
        "sass_architecture": f"sm_{os.environ['COMPUTE_CAPABILITY']}",
        "ptx_embedded": False,
    },
    "test_totals": {"passed": 45, "failed": 0, "skipped": 0},
    "suite_totals": {
        "native_kernels": 24,
        "native_bindings": 15,
        "quantization": 4,
        "lora": 2,
    },
    "kernel_inventory": {
        "total": len(surface["native_kernels"]),
        "in_scope": len(in_scope),
        "tested": len(kernel_correctness),
        "correctness": kernel_correctness,
        "deferred": deferred,
    },
    "native_binding_inventory": {
        "in_scope": len(bindings),
        "tested": len(binding_correctness),
        "correctness": binding_correctness,
    },
    "quantization": {
        "nvfp4": "packed_scales_dequant_scaled_mm_pass",
        "blockwise_fp8": "quant_dequant_gelu_norm_rms_fma_pass",
        "rowwise_int8": "pass",
        "fp6_pack_unpack": "exact_pass",
    },
    "lora": {
        "nvfp4_fuse": "pass",
        "nvfp4_unfuse": "exact_original_snapshot_pass",
        "bf16_regression": "covered_by_M2_storage",
    },
    "numerical_tolerances": {
        "fp32": {**manifest["tolerances"]["fp32"], "result": "pass"},
        "bf16_fp8": {**manifest["tolerances"]["bf16_fp8"], "result": "pass"},
    },
    "output": os.environ["M4_OUTPUT"],
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$m4_output"
