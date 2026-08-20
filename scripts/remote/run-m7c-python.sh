#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkpoint=/workspace/ltx-model-cache/M7A/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors
upsampler=/workspace/ltx-model-cache/M7C/ltx23-spatial-upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
gemma_root=/workspace/ltx-model-cache/M7C/ltx23-gemma3-assets
output=${1:-$repo_root/build/M7C/videos/m7c-python-beach-volleyball.mp4}
offload=${M7C_PYTHON_OFFLOAD:-none}
run_dir="$repo_root/build/M7C/python-run"
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

for required in "$checkpoint" "$upsampler" "$gemma_root/config.json"; do
    [[ -s $required ]] || { echo "missing M7C Python input: $required" >&2; exit 1; }
done

case "$offload" in
    none|cpu|disk) ;;
    *) echo "invalid M7C_PYTHON_OFFLOAD: $offload" >&2; exit 2 ;;
esac

cd "$repo_root"
mkdir -p "$run_dir" "$(dirname "$output")"
: >"$gpu_samples"
scripts/remote/sample-gpu.sh "$gpu_samples" 1 &
sampler_pid=$!
export PYTORCH_ALLOC_CONF=expandable_segments:True

date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/started-at.txt"
printf '%s\n' "$offload" >"$run_dir/offload-mode.txt"
started_epoch=$(date +%s.%N)
.venv/bin/python -m ltx_pipelines.distilled \
    --distilled-checkpoint-path "$checkpoint" \
    --gemma-root "$gemma_root" \
    --spatial-upsampler-path "$upsampler" \
    --prompt "$prompt" \
    --seed 20260821 \
    --num-frames 241 \
    --frame-rate 24 \
    --height 1024 \
    --width 1536 \
    --offload "$offload" \
    --output-path "$output" \
    >"$run_dir/stdout.log" \
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
