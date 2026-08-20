#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M6/low-vram-training.json}
fixture="$repo_root/tests/fixtures/M6"
cd "$repo_root"

scripts/remote/prepare-m6-python.sh
fixture_temp=$(mktemp -d /tmp/ltx-m6-fixture.XXXXXX)
runner_temp=$(mktemp -d /tmp/ltx-m6-runner.XXXXXX)
gpu_samples="$runner_temp/gpu.jsonl"
sampler_pid=
cleanup() {
    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    if [[ $fixture_temp == /tmp/ltx-m6-fixture.* && -d $fixture_temp ]]; then
        rm -rf -- "$fixture_temp"
    fi
    if [[ $runner_temp == /tmp/ltx-m6-runner.* && -d $runner_temp ]]; then
        rm -rf -- "$runner_temp"
    fi
}
trap cleanup EXIT

"$repo_root/.venv/bin/python" scripts/parity/generate-m6-fixtures.py "$fixture_temp"
tracked_files=$(find "$fixture" -maxdepth 1 -type f -printf '%f\n' | sort)
generated_files=$(find "$fixture_temp" -maxdepth 1 -type f -printf '%f\n' | sort)
if [[ $tracked_files != "$generated_files" ]]; then
    echo "tracked M6 fixture file set does not match the pinned Python oracle" >&2
    exit 1
fi
while IFS= read -r name; do
    if ! cmp -s "$fixture/$name" "$fixture_temp/$name"; then
        echo "tracked M6 fixture '$name' does not match the pinned Python oracle" >&2
        exit 1
    fi
done <<<"$tracked_files"

torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
scripts/remote/sample-gpu.sh "$gpu_samples" 1 &
sampler_pid=$!
m6_output=$(dotnet run \
    --project tests/Ltx.TrainerTests/Ltx.TrainerTests.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture" "$torchsharp_library" "$runner_temp/preprocessed")
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
if [[ $m6_output != *"M6 low-VRAM training fixtures: 21 passed, 0 failed; preprocessing=5, training=11, low_vram=5"* ]]; then
    echo "M6 C# runner did not report the complete twenty-one-check pass" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" M6_OUTPUT="$m6_output" GPU_SAMPLES="$gpu_samples" \
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

samples_path = Path(os.environ["GPU_SAMPLES"])
samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
if not samples:
    raise SystemExit("M6 GPU sampler recorded no samples")
peak_memory = max(sample["memory_used_mib"] for sample in samples)
total_memory = min(sample["memory_total_mib"] for sample in samples)
if peak_memory >= total_memory:
    raise SystemExit("M6 low-VRAM run exhausted the visible GPU")

expected = manifest["training"]["expected"]
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
        "safetensors": "0.6.2",
        "cuda": "13.2",
    },
    "test_totals": {"passed": 21, "failed": 0, "skipped": 0},
    "suite_totals": {"preprocessing": 5, "training": 11, "low_vram": 5},
    "preprocessing": {
        "format": "safetensors",
        "video_latents": "projection_and_geometry_pass",
        "text_conditions": "projection_and_attention_mask_pass",
        "dataset_pairing": "strict_sample_and_metadata_validation_pass",
        "overwrite_guard": "pass",
    },
    "training": {
        "mode": "single_gpu_lora_one_step",
        "objective": "flow_matching_velocity_masked_mse",
        "loss": expected["loss"],
        "lora_a_gradient": "deterministic_bf16_pass",
        "lora_b_gradient": "deterministic_bf16_pass",
        "optimizer_step": "cpu_offloaded_adamw8bit_pass",
    },
    "low_vram_profile": {
        **manifest["training"]["options"],
        "base_weight_residence": "cpu_int8",
        "optimizer_state_residence": "cpu_int8",
        "visible_cuda_devices": 1,
        "gpu_peak_memory_mib": peak_memory,
        "gpu_total_memory_mib": total_memory,
    },
    "numerical_tolerances": {
        "fp32": {**manifest["tolerances"]["fp32"], "result": "pass"},
        "bf16": {**manifest["tolerances"]["bf16"], "result": "pass"},
    },
    "output": os.environ["M6_OUTPUT"],
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$m6_output"
