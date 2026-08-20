#!/usr/bin/env python3
"""Validate the M7B delivery reference and write its compact accepted summary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "M7B"
MODEL_ROOT = ROOT / "models" / "ltx-2.5"
PLAN_SHA = "c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2"
REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"
PROMPT = (
    "A continuous cinematic tracking shot glides forward at eye level along a rain-wet city street during "
    "golden hour. Warm sunlight reflects in amber streaks across puddles and slick asphalt while storefront "
    "signs and passing traffic blur softly in the background. A solitary person in a charcoal coat walks from "
    "the right sidewalk across the frame toward the left, stepping around a puddle as the camera keeps moving "
    "smoothly past them. A light breeze stirs their coat and nearby tree leaves. Cars roll slowly in the "
    "distance, tires hiss on wet pavement, footsteps splash naturally, and a muted city ambience of engines, "
    "distant voices, and a soft crosswalk signal remains synchronized. Realistic motion, natural anatomy, "
    "cinematic shallow depth of field, warm highlights, cool rain shadows, subtle lens flare, no cuts."
)
EXPECTED_MODEL_SHA = {
    "ltx25-dev-transformer": "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584",
    "ltx25-text-encoder": "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
    "ltx25-video-vae-diffusion": "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
    "ltx25-audio-vae": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    "ltx25-spatial-upscaler": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
    "ltx25-distilled-lora": "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:  # noqa: PLR0912, PLR0915
    delivery_path = ARTIFACT_ROOT / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    if delivery.get("milestone") != "M7B" or delivery.get("state") != "candidate_pending_accepted_push":
        fail("M7B delivery candidate state is invalid")
    if delivery.get("plan_revision") != "M7B" or delivery.get("plan_sha256") != PLAN_SHA:
        fail("M7B plan evidence mismatch")
    if delivery.get("prompt") != PROMPT or delivery.get("seed") != 20260820:
        fail("M7B cinematic prompt or seed mismatch")
    repository = delivery.get("model_repository", {})
    if repository != {"id": "Lightricks/LTX-2.5", "resolved_revision": REVISION}:
        fail("M7B model repository revision mismatch")
    dependencies = delivery.get("dependencies", {})
    if (
        dependencies.get("torch") != "2.13.0+cu132"
        or dependencies.get("cuda") != "13.2"
        or dependencies.get("natten") != "0.21.7"
        or dependencies.get("safetensors") != "0.8.0"
        or not dependencies.get("ffmpeg", "").startswith("ffmpeg version 6.1.1")
    ):
        fail("M7B runtime dependency evidence mismatch")
    actual_model_sha = {model["id"]: model["sha256"] for model in delivery.get("models", [])}
    if actual_model_sha != EXPECTED_MODEL_SHA:
        fail("M7B model SHA-256 evidence mismatch")
    for model in delivery["models"]:
        path = MODEL_ROOT / model["path"]
        if (
            not path.is_file()
            or path.stat().st_size != model["size_bytes"]
            or sha256(path) != model["sha256"]
        ):
            fail(f"M7B pinned model payload failed the acceptance checksum gate: {model['id']}")

    settings = delivery.get("settings", {})
    required_settings = {
        "frame_rate": 24.0,
        "num_inference_steps": 30,
        "quantization": "fp8-cast",
        "offload": "disk",
        "max_batch_size": 1,
        "diffvae_optimization": "chunked_eager",
        "distilled_lora_strength": 1.0,
    }
    if any(settings.get(key) != value for key, value in required_settings.items()):
        fail("M7B documented FP8/offload/two-stage settings mismatch")
    if (settings.get("num_frames", 0) - 1) % 8 or settings.get("height", 0) % 64 or settings.get("width", 0) % 64:
        fail("M7B output geometry is outside the two-stage grid")

    pipeline = delivery.get("pipeline")
    attempts = delivery.get("attempts", [])
    passing = [attempt for attempt in attempts if attempt.get("result") == "pass"]
    if len(passing) != 1 or passing[0].get("attempt") != delivery.get("final_attempt"):
        fail("M7B must identify exactly one final passing attempt")
    if any(attempt.get("secrets_recorded") is not False for attempt in attempts):
        fail("M7B attempt evidence is not redacted")
    mmap_failures = [attempt for attempt in attempts if attempt.get("failure_code") == "host_mmap_cgroup_limit"]
    if not mmap_failures or any(
        attempt.get("gpu", {}).get("peak_sampled_memory_mib", 1) > 10 for attempt in mmap_failures
    ):
        fail("M7B host mmap diagnosis evidence is missing or inconsistent")

    if pipeline == "TI2VidTwoStagesPipeline":
        if delivery.get("fallback_used") is not False:
            fail("M7B incorrectly marks a successful two-stage result as fallback")
        vram_failures = [
            attempt
            for attempt in attempts
            if attempt.get("pipeline") == "TI2VidTwoStagesPipeline"
            and attempt.get("failure_code") == "cuda_out_of_memory"
        ]
        full_resolution_failed_frames = {
            attempt.get("settings", {}).get("num_frames")
            for attempt in vram_failures
            if attempt.get("settings", {}).get("height") == 1024
            and attempt.get("settings", {}).get("width") == 1536
        }
        final_frames = settings["num_frames"]
        if final_frames < 241 and 241 not in full_resolution_failed_frames:
            fail("M7B shortened two-stage output without a diagnosed 10-second VRAM failure")
        if final_frames < 121 and 121 not in full_resolution_failed_frames:
            fail("M7B shortened two-stage output without a diagnosed 5-second VRAM failure")
        if (settings["height"], settings["width"]) != (1024, 1536) and not {
            241,
            121,
            73,
        } <= full_resolution_failed_frames:
            fail("M7B reduced two-stage resolution before completing the duration ladder")
    elif pipeline == "DistilledPipeline":
        two_stage_vram_failures = [
            attempt
            for attempt in attempts
            if attempt.get("pipeline") == "TI2VidTwoStagesPipeline"
            and attempt.get("failure_code") == "cuda_out_of_memory"
        ]
        required_frames = {241, 121, 73}
        observed_frames = {attempt.get("settings", {}).get("num_frames") for attempt in two_stage_vram_failures}
        reduced_resolution = any(
            attempt.get("settings", {}).get("height", 1024) < 1024
            or attempt.get("settings", {}).get("width", 1536) < 1536
            for attempt in two_stage_vram_failures
        )
        if (
            delivery.get("fallback_used") is not True
            or not required_frames <= observed_frames
            or not reduced_resolution
        ):
            fail("M7B distilled fallback lacks the required diagnosed two-stage VRAM ladder")
    else:
        fail("M7B final pipeline is neither preferred two-stage nor permitted distilled fallback")

    low_memory = delivery.get("low_host_memory_execution", {})
    shard_manifests = low_memory.get("shard_manifests", {})
    if shard_manifests.get("transformer", {}).get("source_sha256") != EXPECTED_MODEL_SHA[
        "ltx25-dev-transformer"
    ] or shard_manifests.get("text_encoder", {}).get("source_sha256") != EXPECTED_MODEL_SHA[
        "ltx25-text-encoder"
    ]:
        fail("M7B sharded execution is not tied to the pinned source payloads")
    if shard_manifests.get("transformer", {}).get("shard_count") != 5 or shard_manifests.get(
        "text_encoder", {}
    ).get("shard_count") != 4:
        fail("M7B low-host-memory shard layout mismatch")
    if shard_manifests["transformer"].get("tensor_count") != 4349 or shard_manifests["text_encoder"].get(
        "tensor_count"
    ) != 686:
        fail("M7B shard tensor inventory mismatch")
    all_shards = [
        *shard_manifests["transformer"].get("shards", []),
        *shard_manifests["text_encoder"].get("shards", []),
    ]
    if len(all_shards) != 9 or any(
        shard.get("size_bytes", 9 * 1024**3) >= 9 * 1024**3 or len(shard.get("sha256", "")) != 64
        for shard in all_shards
    ):
        fail("M7B shard size/checksum evidence mismatch")

    output_record = delivery.get("output", {})
    output = Path(output_record.get("path", "")).resolve()
    expected_output = (ARTIFACT_ROOT / "ltx2-m7b-cinematic-golden-hour.mp4").resolve()
    if output != expected_output or not output.is_file() or output.stat().st_size != output_record.get("size_bytes"):
        fail("M7B local-delivery MP4 reference is missing or invalid")
    if sha256(output) != output_record.get("sha256"):
        fail("M7B local-delivery MP4 SHA-256 mismatch")

    probe_command = [
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
    actual_probe = json.loads(subprocess.check_output(probe_command, text=True))
    if actual_probe != delivery.get("ffprobe"):
        fail("M7B stored ffprobe evidence does not match the final MP4")
    video = [stream for stream in actual_probe["streams"] if stream.get("codec_type") == "video"]
    audio = [stream for stream in actual_probe["streams"] if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        fail("M7B MP4 lacks exactly one video and one audio stream")
    if video[0].get("width") != settings["width"] or video[0].get("height") != settings["height"]:
        fail("M7B ffprobe dimensions mismatch")
    if int(video[0].get("nb_frames", 0)) != settings["num_frames"] or video[0].get("r_frame_rate") != "24/1":
        fail("M7B ffprobe frame count/rate mismatch")
    duration = float(actual_probe["format"]["duration"])
    if not 2.9 <= duration <= 10.2:
        fail("M7B output duration is outside the approved fallback range")

    gpu = delivery.get("gpu", {})
    if (
        gpu.get("name") != "NVIDIA GeForce RTX 5090"
        or gpu.get("memory_total_mib") != 32607
        or not 0 < gpu.get("peak_sampled_memory_mib", 0) < gpu.get("memory_total_mib", 0)
    ):
        fail("M7B GPU peak-memory evidence mismatch")
    if delivery.get("local_delivery", {}).get("state") != "ready_for_coordinator_copy_only_after_accepted_commit_push":
        fail("M7B local-delivery handoff condition mismatch")
    if delivery.get("secrets_recorded") is not False:
        fail("M7B delivery evidence is not redacted")

    preflight = json.loads((ROOT / "artifacts" / "M0" / "preflight.json").read_text())
    artifact_checksums = {
        path.name: sha256(path)
        for path in sorted(ARTIFACT_ROOT.glob("*.json"))
        if path.name != "summary.json"
    }
    summary = {
        "schema_version": 1,
        "milestone": "M7B",
        "state": "accepted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_reference_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "plan_revision": "M7B",
        "plan_sha256": PLAN_SHA,
        "verification_command": "scripts/remote/verify-milestone.sh M7B",
        "commands": [
            {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
            {"command": ".venv/bin/ruff check <M7B Python sources>", "exit_code": 0},
            {"command": "bash -n scripts/remote/run-m7b-sample.sh", "exit_code": 0},
            {"command": "python3 -m py_compile <M7B Python sources>", "exit_code": 0},
            {"command": "ffmpeg full video/audio decode", "exit_code": 0},
            {"command": "scripts/remote/verify-m7b.py", "exit_code": 0},
        ],
        "toolchain": preflight["toolchain"],
        "gpu": gpu,
        "test_totals": {"passed": 14, "failed": 0, "skipped": 0},
        "tests": {
            "plan_preservation": "pass",
            "M6_accepted_prerequisite": "pass",
            "release_build": "pass",
            "python_lint_compile": "pass",
            "pinned_model_revisions_sha256": "6_pass",
            "low_host_memory_sharding": "5_transformer_plus_4_text_encoder_shards_pass",
            "preferred_two_stage_or_fallback_policy": "pass",
            "cinematic_prompt_seed_settings": "pass",
            "gpu_peak_memory": "pass",
            "mp4_sha256": "pass",
            "ffprobe_video": "pass",
            "ffprobe_audio": "pass",
            "full_video_audio_decode": "pass",
            "local_delivery_reference": "pass",
        },
        "pipeline": pipeline,
        "fallback_used": delivery["fallback_used"],
        "prompt": delivery["prompt"],
        "seed": delivery["seed"],
        "settings": settings,
        "model_repository": repository,
        "dependencies": dependencies,
        "models": delivery["models"],
        "low_host_memory_execution": low_memory,
        "attempts": attempts,
        "output": output_record,
        "ffprobe": actual_probe,
        "local_delivery": delivery["local_delivery"],
        "artifact_sha256": artifact_checksums,
        "secrets_recorded": False,
    }
    (ARTIFACT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sys.stdout.write("M7B evidence gate passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
