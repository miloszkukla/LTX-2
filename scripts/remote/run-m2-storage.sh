#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M2/storage.json}
fixture="$repo_root/tests/fixtures/M2"
cd "$repo_root"

scripts/remote/prepare-m2-python.sh
fixture_temp=$(mktemp -d /tmp/ltx-m2-fixture.XXXXXX)
output_temp=$(mktemp -d /tmp/ltx-m2-output.XXXXXX)
cleanup() {
    if [[ $fixture_temp == /tmp/ltx-m2-fixture.* && -d $fixture_temp ]]; then
        rm -rf -- "$fixture_temp"
    fi
    if [[ $output_temp == /tmp/ltx-m2-output.* && -d $output_temp ]]; then
        rm -rf -- "$output_temp"
    fi
}
trap cleanup EXIT

"$repo_root/.venv/bin/python" scripts/parity/generate-m2-fixtures.py "$fixture_temp"
tracked_files=$(find "$fixture" -maxdepth 1 -type f -printf '%f\n' | sort)
generated_files=$(find "$fixture_temp" -maxdepth 1 -type f -printf '%f\n' | sort)
if [[ $tracked_files != "$generated_files" ]]; then
    echo "tracked M2 fixture file set does not match the pinned Python oracle" >&2
    exit 1
fi
while IFS= read -r name; do
    if ! cmp -s "$fixture/$name" "$fixture_temp/$name"; then
        echo "tracked M2 fixture '$name' does not match the pinned Python oracle" >&2
        exit 1
    fi
done <<<"$tracked_files"

torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
storage_output=$(dotnet run \
    --project tests/Ltx.StorageTests/Ltx.StorageTests.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture" "$output_temp" "$torchsharp_library")
if [[ $storage_output != *"M2 storage fixtures: 13 passed, 0 failed"* ]]; then
    echo "M2 C# storage runner did not report thirteen passing fixtures" >&2
    exit 1
fi

python_output=$("$repo_root/.venv/bin/python" \
    scripts/parity/verify-m2-roundtrip.py "$fixture" "$output_temp")
PYTHON_OUTPUT="$python_output" "$repo_root/.venv/bin/python" - <<'PY'
import json
import os

result = json.loads(os.environ["PYTHON_OUTPUT"])
if result != {
    "checks": 3,
    "message": "Python safetensors accepted all C# round trips with exact layouts, payloads, and metadata.",
    "result": "pass",
}:
    raise SystemExit(f"unexpected M2 Python interop result: {result}")
PY

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" STORAGE_OUTPUT="$storage_output" PYTHON_OUTPUT="$python_output" \
"$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

fixture = Path(os.environ["FIXTURE"])
manifest = json.loads((fixture / "manifest.json").read_text())
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

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "fixture_revision": manifest["fixture_revision"],
    "fixture_set_sha256": digest.hexdigest(),
    "fixture_sha256": checksums,
    "oracle": manifest["oracle"],
    "dependencies": {
        "numpy": "2.3.2",
        "packaging": "25.0",
        "safetensors": "0.6.2",
        "torchsharp": "0.107.0",
    },
    "test_totals": {"passed": 16, "failed": 0, "skipped": 0},
    "tests": {
        "csharp_storage_and_loader_fixtures": "13_pass",
        "python_roundtrip_interop": "3_pass",
        "safetensors_round_trip": "exact",
        "lora_round_trip": "exact",
        "checkpoint_shards": "pass",
        "metadata_json": "pass",
        "lora_key_mapping": "pass",
        "lora_fuse_unfuse": "pass",
    },
    "numerical_tolerances": {
        "fp32": {**manifest["tolerances"]["fp32"], "result": "pass"},
        "bf16": {**manifest["tolerances"]["bf16"], "result": "pass"},
    },
    "round_trip": {
        "safetensors": "exact_dtype_shape_payload_metadata",
        "lora": "exact_dtype_shape_payload_metadata",
    },
    "output": os.environ["STORAGE_OUTPUT"],
    "python_output": json.loads(os.environ["PYTHON_OUTPUT"]),
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$storage_output"
printf '%s\n' "$python_output"
