#!/usr/bin/env bash
set -Eeuo pipefail

attempt=${1:?usage: run-m7b-sample.sh ATTEMPT NUM_FRAMES HEIGHT WIDTH OUTPUT.mp4}
num_frames=${2:?usage: run-m7b-sample.sh ATTEMPT NUM_FRAMES HEIGHT WIDTH OUTPUT.mp4}
height=${3:?usage: run-m7b-sample.sh ATTEMPT NUM_FRAMES HEIGHT WIDTH OUTPUT.mp4}
width=${4:?usage: run-m7b-sample.sh ATTEMPT NUM_FRAMES HEIGHT WIDTH OUTPUT.mp4}
output=${5:?usage: run-m7b-sample.sh ATTEMPT NUM_FRAMES HEIGHT WIDTH OUTPUT.mp4}

if [[ ! $attempt =~ ^[a-z0-9][a-z0-9-]{0,63}$ || \
      ! $num_frames =~ ^[1-9][0-9]*$ || \
      ! $height =~ ^[1-9][0-9]*$ || \
      ! $width =~ ^[1-9][0-9]*$ ]]; then
    echo "attempt must be a short lowercase identifier and dimensions must be positive integers" >&2
    exit 2
fi
if (( (num_frames - 1) % 8 != 0 || height % 64 != 0 || width % 64 != 0 )); then
    echo "two-stage geometry requires 8k+1 frames and dimensions divisible by 64" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

transformer=models/ltx-2.5/diffusion_models/dev-sharded
text_encoder=models/ltx-2.5/text_encoders/gemma4-sharded
video_vae=models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors
audio_vae=models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
spatial_upsampler=models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
distilled_lora=models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
for required in "$transformer" "$text_encoder" "$video_vae" "$audio_vae" "$spatial_upsampler" "$distilled_lora"; do
    if [[ ! -e $required ]]; then
        echo "required pinned model payload is missing: $required" >&2
        exit 1
    fi
done

output=$(realpath -m "$output")
if [[ ${output##*.} != mp4 ]]; then
    echo "output must use the .mp4 extension" >&2
    exit 2
fi
if [[ -e $output ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 1
fi

work_dir="$repo_root/tmp/M7B/$attempt"
artifact="$repo_root/artifacts/M7B/$attempt.json"
if [[ -e $work_dir || -e $artifact ]]; then
    echo "refusing to overwrite existing attempt evidence: $attempt" >&2
    exit 1
fi
mkdir -p "$work_dir" "$(dirname "$output")" "$repo_root/artifacts/M7B"

prompt="A continuous cinematic tracking shot glides forward at eye level along a rain-wet city street during golden hour. Warm sunlight reflects in amber streaks across puddles and slick asphalt while storefront signs and passing traffic blur softly in the background. A solitary person in a charcoal coat walks from the right sidewalk across the frame toward the left, stepping around a puddle as the camera keeps moving smoothly past them. A light breeze stirs their coat and nearby tree leaves. Cars roll slowly in the distance, tires hiss on wet pavement, footsteps splash naturally, and a muted city ambience of engines, distant voices, and a soft crosswalk signal remains synchronized. Realistic motion, natural anatomy, cinematic shallow depth of field, warm highlights, cool rain shadows, subtle lens flare, no cuts."
seed=20260820
frame_rate=24
metrics="$work_dir/gpu.jsonl"
log="$work_dir/pipeline.log"

start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
scripts/remote/sample-gpu.sh "$metrics" 1 &
sampler_pid=$!
stop_sampler() {
    if kill -0 "$sampler_pid" 2>/dev/null; then
        kill "$sampler_pid" 2>/dev/null || true
    fi
    wait "$sampler_pid" 2>/dev/null || true
}
trap stop_sampler EXIT

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m ltx_pipelines.ti2vid_two_stages \
    --transformer-path "$transformer" \
    --text-encoder-path "$text_encoder" \
    --video-vae-path "$video_vae" \
    --audio-vae-path "$audio_vae" \
    --spatial-upsampler-path "$spatial_upsampler" \
    --distilled-lora "$distilled_lora" 1.0 \
    --prompt "$prompt" \
    --seed "$seed" \
    --num-frames "$num_frames" \
    --height "$height" \
    --width "$width" \
    --frame-rate "$frame_rate" \
    --num-inference-steps 30 \
    --video-cfg-guidance-scale 3.0 \
    --video-stg-guidance-scale 1.0 \
    --video-rescale-scale 0.7 \
    --video-stg-blocks 28 \
    --a2v-guidance-scale 3.0 \
    --video-skip-step 0 \
    --audio-cfg-guidance-scale 7.0 \
    --audio-stg-guidance-scale 1.0 \
    --audio-rescale-scale 0.7 \
    --audio-stg-blocks 28 \
    --v2a-guidance-scale 3.0 \
    --audio-skip-step 0 \
    --quantization fp8-cast \
    --offload disk \
    --max-batch-size 1 \
    --diffvae-optimization chunked_eager \
    --num-generated-keyframes 0 \
    --output-path "$output" \
    >"$log" 2>&1
exit_code=$?
set -e

stop_sampler
trap - EXIT
finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
finish_epoch=$(date +%s)
runtime_seconds=$((finish_epoch - start_epoch))

ATTEMPT="$attempt" \
NUM_FRAMES="$num_frames" \
HEIGHT="$height" \
WIDTH="$width" \
OUTPUT="$output" \
PROMPT="$prompt" \
SEED="$seed" \
FRAME_RATE="$frame_rate" \
EXIT_CODE="$exit_code" \
START_UTC="$start_utc" \
FINISH_UTC="$finish_utc" \
RUNTIME_SECONDS="$runtime_seconds" \
METRICS="$metrics" \
LOG="$log" \
ARTIFACT="$artifact" \
python3 - <<'PY'
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

metrics_path = Path(os.environ["METRICS"])
samples = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
peak = max((sample["memory_used_mib"] for sample in samples), default=0)
gpu_name, gpu_total = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().rsplit(",", 1)

exit_code = int(os.environ["EXIT_CODE"])
output = Path(os.environ["OUTPUT"])
log_path = Path(os.environ["LOG"])
log_text = log_path.read_text(errors="replace")
if exit_code == 0 and output.is_file() and output.stat().st_size > 0:
    result = "pass"
    failure_code = "none"
elif re.search(r"CUDA out of memory|OutOfMemoryError", log_text, re.IGNORECASE):
    result = "fail"
    failure_code = "cuda_out_of_memory"
elif exit_code == 137 or re.search(r"(?:^|\n)Killed(?:\n|$)", log_text):
    result = "fail"
    failure_code = "host_out_of_memory"
elif exit_code == 0:
    result = "fail"
    failure_code = "missing_output"
else:
    result = "fail"
    failure_code = "pipeline_runtime_error"

ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
interesting = [
    ansi.sub("", line).strip()
    for line in log_text.splitlines()
    if re.search(r"error|exception|traceback|out of memory|killed", line, re.IGNORECASE)
]
excerpt = interesting[-1][:500] if interesting else None

payload = {
    "schema_version": 1,
    "milestone": "M7B",
    "attempt": os.environ["ATTEMPT"],
    "pipeline": "TI2VidTwoStagesPipeline",
    "result": result,
    "failure_code": failure_code,
    "failure_excerpt": excerpt,
    "exit_code": exit_code,
    "started_at_utc": os.environ["START_UTC"],
    "finished_at_utc": os.environ["FINISH_UTC"],
    "runtime_seconds": int(os.environ["RUNTIME_SECONDS"]),
    "prompt": os.environ["PROMPT"],
    "seed": int(os.environ["SEED"]),
    "settings": {
        "num_frames": int(os.environ["NUM_FRAMES"]),
        "target_duration_seconds": (int(os.environ["NUM_FRAMES"]) - 1) / float(os.environ["FRAME_RATE"]),
        "frame_rate": float(os.environ["FRAME_RATE"]),
        "height": int(os.environ["HEIGHT"]),
        "width": int(os.environ["WIDTH"]),
        "num_inference_steps": 30,
        "quantization": "fp8-cast",
        "offload": "disk",
        "max_batch_size": 1,
        "diffvae_optimization": "chunked_eager",
        "distilled_lora_strength": 1.0,
    },
    "gpu": {
        "name": gpu_name.strip(),
        "memory_total_mib": int(gpu_total.strip()),
        "peak_sampled_memory_mib": peak,
        "sample_interval_seconds": 1,
    },
    "output": {
        "path": str(output),
        "exists": output.is_file(),
        "size_bytes": output.stat().st_size if output.is_file() else 0,
    },
    "diagnostic_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    "secrets_recorded": False,
}
Path(os.environ["ARTIFACT"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if (( exit_code != 0 )); then
    echo "M7B attempt $attempt failed; see $artifact" >&2
    exit "$exit_code"
fi
if [[ ! -s $output ]]; then
    echo "M7B attempt $attempt returned success without a playable candidate" >&2
    exit 1
fi
echo "M7B attempt $attempt produced $output"
