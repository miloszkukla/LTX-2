#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M3/core-model.json}
fixture="$repo_root/tests/fixtures/M3"
cd "$repo_root"

scripts/remote/prepare-m3-python.sh
fixture_temp=$(mktemp -d /tmp/ltx-m3-fixture.XXXXXX)
cleanup() {
    if [[ $fixture_temp == /tmp/ltx-m3-fixture.* && -d $fixture_temp ]]; then
        rm -rf -- "$fixture_temp"
    fi
}
trap cleanup EXIT

"$repo_root/.venv/bin/python" scripts/parity/generate-m3-fixtures.py "$fixture_temp"
tracked_files=$(find "$fixture" -maxdepth 1 -type f -printf '%f\n' | sort)
generated_files=$(find "$fixture_temp" -maxdepth 1 -type f -printf '%f\n' | sort)
if [[ $tracked_files != "$generated_files" ]]; then
    echo "tracked M3 fixture file set does not match the pinned Python oracle" >&2
    exit 1
fi
while IFS= read -r name; do
    if [[ $name == manifest.json ]]; then
        "$repo_root/.venv/bin/python" scripts/parity/compare-cross-gpu-fixtures.py \
            --allow-device-name "$fixture/$name" "$fixture_temp/$name"
    elif ! cmp -s "$fixture/$name" "$fixture_temp/$name"; then
        echo "tracked M3 fixture '$name' does not match the pinned Python oracle" >&2
        exit 1
    fi
done <<<"$tracked_files"

torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
core_output=$(dotnet run \
    --project tests/Ltx.CoreExecutionTests/Ltx.CoreExecutionTests.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture" "$torchsharp_library" "$fixture/offload.safetensors")
if [[ $core_output != *"M3 core fixtures: 34 passed, 0 failed"* ]]; then
    echo "M3 C# core-model runner did not report thirty-four passing fixtures" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" CORE_OUTPUT="$core_output" \
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
        "safetensors": "0.6.2",
        "torch": "2.13.0+cu132",
        "torchsharp": "0.107.0",
    },
    "test_totals": {"passed": 34, "failed": 0, "skipped": 0},
    "suite_totals": {
        "transformer": 7,
        "vae": 8,
        "audio_vocoder": 9,
        "conditioning": 4,
        "offload": 6,
    },
    "tests": {
        "transformer": "seeded_fp32_bf16_pass",
        "vae": "seeded_fp32_bf16_pass",
        "audio_vocoder": "seeded_fp32_bf16_pass",
        "conditioning": "seeded_fp32_pass",
        "offload": "none_cpu_disk_seeded_fp32_bf16_pass",
    },
    "numerical_tolerances": {
        "fp32": {**manifest["tolerances"]["fp32"], "result": "pass"},
        "bf16": {**manifest["tolerances"]["bf16"], "result": "pass"},
    },
    "output": os.environ["CORE_OUTPUT"],
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$core_output"
