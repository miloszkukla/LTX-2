#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_dir=${1:-$repo_root/artifacts/M7A}
build_dir="$repo_root/build/M7A/acceptance"
checkpoint_root=/workspace/ltx-model-cache/M7A
checkpoint="$checkpoint_root/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors"
contexts="$checkpoint_root/ltx23-hdr-assets/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
oracle="$build_dir/real-checkpoint-oracle.safetensors"
torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
runner_temp=$(mktemp -d /tmp/ltx-m7a-runner.XXXXXX)
gpu_samples="$runner_temp/gpu.jsonl"
host_samples="$runner_temp/host-memory-kib.txt"
sampler_pid=
host_sampler_pid=

cleanup() {
    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    if [[ -n $host_sampler_pid ]]; then
        kill "$host_sampler_pid" 2>/dev/null || true
        wait "$host_sampler_pid" 2>/dev/null || true
    fi
    if [[ $runner_temp == /tmp/ltx-m7a-runner.* && -d $runner_temp ]]; then
        rm -rf -- "$runner_temp"
    fi
}
trap cleanup EXIT

cd "$repo_root"
mkdir -p "$artifact_dir" "$build_dir/media"
scripts/remote/download-m7a-models.py \
    --cache-root "$checkpoint_root" \
    --ids ltx23-distilled-checkpoint ltx23-hdr-assets \
    --report "$artifact_dir/model-download.json"
scripts/remote/prepare-m3-python.sh
scripts/remote/build-m5-media.sh >/dev/null

"$repo_root/.venv/bin/python" - <<'PY'
from ltx_core.loader import SingleGPUModelBuilder
from ltx_core.model.transformer import LTXModelConfigurator
from ltx_core.model.video_vae import VideoDecoderConfigurator
from ltx_core.model.audio_vae import AudioDecoderConfigurator, VocoderConfigurator
PY

scripts/remote/sample-gpu.sh "$gpu_samples" 1 &
sampler_pid=$!
(
    while true; do
        awk '/^MemAvailable:/ {print $2}' /proc/meminfo >>"$host_samples"
        sleep 1
    done
) &
host_sampler_pid=$!

"$repo_root/.venv/bin/python" scripts/fixtures/generate-m7a-checkpoint-oracle.py \
    --checkpoint "$checkpoint" \
    --contexts "$contexts" \
    --output "$oracle"
cp "$build_dir/real-checkpoint-oracle.json" "$artifact_dir/checkpoint-oracle.json"

library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M5/ltx-media${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
dotnet run \
    --project tests/Ltx.CheckpointRuntimeTests/Ltx.CheckpointRuntimeTests.csproj \
    --configuration Release \
    --no-build \
    -- "$checkpoint" "$oracle" "$torchsharp_library" "$artifact_dir/checkpoint-runtime.json"
dotnet run \
    --project tests/Ltx.RealPipelineTests/Ltx.RealPipelineTests.csproj \
    --configuration Release \
    --no-build \
    -- "$checkpoint" "$contexts" "$torchsharp_library" "$build_dir/media" "$artifact_dir/real-pipelines.json"
dotnet run \
    --project tests/Ltx.StandardTrainerTests/Ltx.StandardTrainerTests.csproj \
    --configuration Release \
    --no-build \
    -- "$checkpoint" "$oracle" "$torchsharp_library" "$artifact_dir/standard-trainer.json"

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
kill "$host_sampler_pid" 2>/dev/null || true
wait "$host_sampler_pid" 2>/dev/null || true
host_sampler_pid=

REPORT="$artifact_dir/hardware.json" GPU_SAMPLES="$gpu_samples" HOST_SAMPLES="$host_samples" \
"$repo_root/.venv/bin/python" - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
if len(query) != 1:
    raise SystemExit("M7A requires exactly one visible GPU")
name, capability, driver, memory = [part.strip() for part in query[0].split(",")]
gpu_samples = [
    json.loads(line)
    for line in Path(os.environ["GPU_SAMPLES"]).read_text().splitlines()
    if line.strip()
]
host_samples = [
    int(line)
    for line in Path(os.environ["HOST_SAMPLES"]).read_text().splitlines()
    if line.strip()
]
if not gpu_samples or not host_samples:
    raise SystemExit("M7A hardware sampler recorded no samples")
report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "gpu": {
        "name": name,
        "compute_capability": capability,
        "driver": driver,
        "device_count": 1,
        "memory_total_mib": int(memory),
        "peak_memory_used_mib": max(item["memory_used_mib"] for item in gpu_samples),
        "peak_temperature_c": max(item["temperature_c"] for item in gpu_samples),
    },
    "host": {
        "minimum_memory_available_mib": min(host_samples) // 1024,
    },
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
