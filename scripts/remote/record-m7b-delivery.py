#!/usr/bin/env python3
"""Record the redacted M7B delivery candidate and its local-copy reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "M7B"
MODEL_ROOT = ROOT / "models" / "ltx-2.5"
REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"
EXPECTED_MODELS = {
    "ltx25-dev-transformer": (
        "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        42018190584,
        "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584",
    ),
    "ltx25-text-encoder": (
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        26263858182,
        "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
    ),
    "ltx25-video-vae-diffusion": (
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        1472223346,
        "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
    ),
    "ltx25-audio-vae": (
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        364866540,
        "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    ),
    "ltx25-spatial-upscaler": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        995778752,
        "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
    ),
    "ltx25-distilled-lora": (
        "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        8899889568,
        "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", help="Successful attempt artifact name, without .json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    attempt_path = ARTIFACT_ROOT / f"{args.attempt}.json"
    attempt = json.loads(attempt_path.read_text())
    output = args.output.resolve()
    if attempt.get("result") != "pass" or attempt.get("output", {}).get("path") != str(output):
        raise SystemExit("selected final attempt is not a passing attempt for this output")
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit("delivery MP4 is missing or empty")

    model_records = []
    for model_id, (relative_path, expected_size, expected_sha) in EXPECTED_MODELS.items():
        path = MODEL_ROOT / relative_path
        actual_sha = sha256(path)
        if path.stat().st_size != expected_size or actual_sha != expected_sha:
            raise SystemExit(f"pinned model checksum/size mismatch: {model_id}")
        model_records.append(
            {
                "id": model_id,
                "path": relative_path,
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )

    transformer_shards = MODEL_ROOT / "diffusion_models" / "dev-sharded" / "shards.json"
    text_encoder_shards = MODEL_ROOT / "text_encoders" / "gemma4-sharded" / "shards.json"
    shard_manifests = {}
    for name, path in (("transformer", transformer_shards), ("text_encoder", text_encoder_shards)):
        manifest = json.loads(path.read_text())
        shard_manifests[name] = {
            "shard_count": manifest["shard_count"],
            "tensor_count": manifest["tensor_count"],
            "source_size_bytes": manifest["source_size_bytes"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": sha256(path),
            "shards": manifest["shards"],
        }

    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,duration,size,bit_rate:"
            "stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,"
            "avg_frame_rate,nb_frames,sample_rate,channels,duration"
        ),
        "-of",
        "json",
        str(output),
    ]
    probe = json.loads(command(*ffprobe_command))
    video_streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise SystemExit("delivery MP4 must contain exactly one video and one audio stream")
    if video_streams[0].get("width") != attempt["settings"]["width"] or video_streams[0].get(
        "height"
    ) != attempt["settings"]["height"]:
        raise SystemExit("ffprobe dimensions do not match the generation settings")

    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if decode.returncode != 0:
        raise SystemExit("full MP4 video/audio decode failed")

    python_dependencies = json.loads(
        command(
            str(ROOT / ".venv" / "bin" / "python"),
            "-c",
            (
                "import json,natten,safetensors,torch; "
                "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'natten':natten.__version__,'safetensors':safetensors.__version__}))"
            ),
        )
    )

    attempts = []
    for path in sorted(ARTIFACT_ROOT.glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("pipeline") in {"TI2VidTwoStagesPipeline", "DistilledPipeline"}:
            attempts.append(value)

    delivery = {
        "schema_version": 1,
        "milestone": "M7B",
        "state": "candidate_pending_accepted_push",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_reference_sha": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "plan_revision": "M7B",
        "plan_sha256": "c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2",
        "pipeline": attempt["pipeline"],
        "fallback_used": attempt["pipeline"] == "DistilledPipeline",
        "prompt": attempt["prompt"],
        "seed": attempt["seed"],
        "settings": attempt["settings"],
        "model_repository": {"id": "Lightricks/LTX-2.5", "resolved_revision": REVISION},
        "models": model_records,
        "dependencies": {
            **python_dependencies,
            "ffmpeg": command("ffmpeg", "-version").splitlines()[0],
        },
        "low_host_memory_execution": {
            "reason": "Published single files exceeded the 31,403,802,624-byte host-memory cgroup mmap limit.",
            "method": "byte-identical safetensors shards with one active disk-streaming mmap",
            "shard_manifests": shard_manifests,
        },
        "attempts": attempts,
        "final_attempt": args.attempt,
        "gpu": attempt["gpu"],
        "output": {
            "path": str(output),
            "filename": output.name,
            "size_bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "ffprobe": probe,
        "ffprobe_command": " ".join([*ffprobe_command[:-1], "<delivery.mp4>"]),
        "full_decode": {"command": "ffmpeg -v error -i <delivery.mp4> -map 0:v:0 -map 0:a:0 -f null -", "exit_code": 0},
        "local_delivery": {
            "state": "ready_for_coordinator_copy_only_after_accepted_commit_push",
            "copy_command": f"copy remote:{output} with recorded SHA-256 verification",
        },
        "secrets_recorded": False,
    }
    (ARTIFACT_ROOT / "delivery.json").write_text(json.dumps(delivery, indent=2, sort_keys=True) + "\n")
    sys.stdout.write("M7B delivery candidate recorded\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
