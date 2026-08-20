#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M1/parity.json}
fixture="$repo_root/tests/fixtures/M1/foundation-fp32.json"
cd "$repo_root"

scripts/remote/prepare-m1-abi.sh
fixture_temp=$(mktemp /tmp/ltx-m1-fixture.XXXXXX.json)
cleanup() {
    if [[ $fixture_temp == /tmp/ltx-m1-fixture.*.json && -f $fixture_temp ]]; then
        rm -f -- "$fixture_temp"
    fi
}
trap cleanup EXIT
"$repo_root/.venv/bin/python" scripts/parity/generate-m1-fixture.py "$fixture_temp"
if ! cmp -s "$fixture" "$fixture_temp"; then
    echo "tracked M1 fixture does not match the pinned Python oracle" >&2
    exit 1
fi

torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
cuda_library="$repo_root/build/M1/ltx-cuda/libltx_cuda.so"
library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M1/ltx-cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
parity_output=$(dotnet run \
    --project tests/Ltx.ParityRunner/Ltx.ParityRunner.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture" "$torchsharp_library" "$cuda_library")
if [[ $parity_output != *"Parity fixtures: 2 passed, 0 failed"* ]]; then
    echo "M1 parity runner did not report two passing fixtures" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" PARITY_OUTPUT="$parity_output" \
"$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

fixture = Path(os.environ["FIXTURE"])
report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "fixture_revision": "m1-foundation-v1",
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "oracle": {"pytorch": "2.13.0+cu132", "cuda": "13.2", "seed": 20260820},
    "test_totals": {"passed": 2, "failed": 0, "skipped": 0},
    "numerical_tolerances": {"fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"}},
    "output": os.environ["PARITY_OUTPUT"],
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$parity_output"
