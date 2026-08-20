#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkpoint=/workspace/ltx-model-cache/M7A/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors
upsampler=/workspace/ltx-model-cache/M7C/ltx23-spatial-upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
contexts="$repo_root/build/M7C/beach-volleyball-context.safetensors"
output=${1:-$repo_root/build/M7C/videos/m7c-csharp-beach-volleyball.mp4}
run_dir="$repo_root/build/M7C/csharp-run"
gpu_samples="$run_dir/gpu.jsonl"
sampler_pid=

prompt='Two adult women with long blonde hair play a lively beach-volleyball rally on a sunny beach. They wear sporty bikinis appropriate for beach volleyball and speak casually to each other between plays, smiling and calling the ball. Natural, realistic athletic motion; ocean waves, warm sand, gentle sea breeze, distant beach ambience, synchronized dialogue and sound. Cinematic tracking camera, no cuts.'

cleanup() {
    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for required in "$checkpoint" "$upsampler" "$contexts"; do
    [[ -s $required ]] || { echo "missing M7C input: $required" >&2; exit 1; }
done

cd "$repo_root"
mkdir -p "$run_dir" "$(dirname "$output")"
: >"$gpu_samples"
scripts/remote/sample-gpu.sh "$gpu_samples" 1 &
sampler_pid=$!

library_path=$(scripts/remote/m1-library-path.sh)
export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M5/ltx-media${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LTX_TORCHSHARP_LIBRARY="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
export PYTORCH_ALLOC_CONF=expandable_segments:True

date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/started-at.txt"
started_epoch=$(date +%s.%N)
dotnet run \
    --project src/Ltx.Cli/Ltx.Cli.csproj \
    --configuration Release \
    --no-build \
    -- \
    ti2vid-two-stages \
    --checkpoint-path "$checkpoint" \
    --spatial-upsampler-path "$upsampler" \
    --text-embeddings "$contexts" \
    --prompt "$prompt" \
    --seed 20260821 \
    --num-frames 241 \
    --frame-rate 24 \
    --height 1024 \
    --width 1536 \
    --num-inference-steps 8 \
    --offload-mode disk \
    --latent-diagnostics "$repo_root/build/M7C/diagnostics/csharp-stage-latents.safetensors" \
    --output-path "$output" \
    >"$run_dir/stdout.json" \
    2>"$run_dir/stderr.log"
finished_epoch=$(date +%s.%N)
awk -v started="$started_epoch" -v finished="$finished_epoch" \
    'BEGIN { printf "%.6f\n", finished - started }' >"$run_dir/elapsed-seconds.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/finished-at.txt"

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
[[ -s $output ]]
echo "$output"
