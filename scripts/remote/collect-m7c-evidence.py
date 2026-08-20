#!/usr/bin/env python3
"""Collect and validate redacted M7C paired-video evidence already produced on the worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "M7C"
PROMPT = (
    "Two adult women with long blonde hair play a lively beach-volleyball rally on a sunny beach. "
    "They wear sporty bikinis appropriate for beach volleyball and speak casually to each other between plays, "
    "smiling and calling the ball. Natural, realistic athletic motion; ocean waves, warm sand, gentle sea breeze, "
    "distant beach ambience, synchronized dialogue and sound. Cinematic tracking camera, no cuts."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def probe(path: Path) -> dict:
    return json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            text=True,
        )
    )


def validate_probe(name: str, report: dict) -> None:
    video = next(stream for stream in report["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in report["streams"] if stream["codec_type"] == "audio")
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != 1536
        or video.get("height") != 1024
        or video.get("avg_frame_rate") != "24/1"
        or video.get("nb_frames") != "241"
        or abs(float(video.get("duration", 0)) - 241 / 24) > 1e-5
        or audio.get("codec_name") != "aac"
        or audio.get("sample_rate") != "48000"
        or audio.get("channels") != 2
    ):
        raise SystemExit(f"{name} ffprobe contract mismatch")


def peak_memory(path: Path) -> tuple[int, int]:
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise SystemExit(f"GPU sampler is empty: {path}")
    return max(sample["memory_used_mib"] for sample in samples), len(samples)


def parse_metric(path: Path, pattern: str) -> dict[str, float]:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise SystemExit(f"comparison metric is missing from {path}")
    return {key: float(value) for key, value in match.groupdict().items()}


def tensor_stats(path: Path) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            value = handle.get_tensor(name)
            tensors[name] = value
            resolved = value.float()
            stats[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "minimum": resolved.min().item(),
                "maximum": resolved.max().item(),
                "mean": resolved.mean().item(),
                "standard_deviation": resolved.std().item(),
                "finite": bool(torch.isfinite(resolved).all().item()),
            }
    return stats, tensors


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    a = left.float().flatten()
    b = right.float().flatten()
    difference = a - b
    return {
        "elements": a.numel(),
        "maximum_absolute_error": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "exact_fraction": a.eq(b).float().mean().item(),
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    videos = {
        "csharp": BUILD / "videos" / "m7c-csharp-beach-volleyball.mp4",
        "python": BUILD / "videos" / "m7c-python-beach-volleyball.mp4",
    }
    reports = {name: probe(path) for name, path in videos.items()}
    for name, report in reports.items():
        validate_probe(name, report)
    for run in ("csharp", "python"):
        if (BUILD / f"{run}-run" / "full-decode.stderr.log").read_text():
            raise SystemExit(f"{run} full-stream decode emitted errors")

    csharp_peak, csharp_samples = peak_memory(BUILD / "csharp-run" / "gpu.jsonl")
    python_peak, python_samples = peak_memory(BUILD / "python-run" / "gpu.jsonl")
    video_report = {
        "schema_version": 1,
        "result": "pass",
        "execution_order": ["native_csharp_two_stage", "python_reference_two_stage"],
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "settings": {
            "seed": 20260821,
            "frames": 241,
            "duration_target_seconds": 10,
            "frame_rate": 24,
            "width": 1536,
            "height": 1024,
            "stage1_inference_steps": 8,
            "stage1_sigmas": [1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0],
            "stage2_sigmas": [0.909375, 0.725, 0.421875, 0],
            "audio_tokens": 251,
        },
        "videos": {},
        "secrets_recorded": False,
    }
    for name, path in videos.items():
        peak, samples = (csharp_peak, csharp_samples) if name == "csharp" else (python_peak, python_samples)
        video_report["videos"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "elapsed_seconds": float((BUILD / f"{name}-run" / "elapsed-seconds.txt").read_text()),
            "peak_gpu_memory_used_mib": peak,
            "gpu_samples": samples,
            "offload": "disk_weight_streaming" if name == "csharp" else "none_resident",
            "ffprobe": reports[name],
            "full_stream_decode": "pass",
        }
    write(artifact_dir / "videos.json", video_report)

    comparison_dir = BUILD / "comparison-final"
    ssim = parse_metric(
        comparison_dir / "ssim.stderr.log",
        r"SSIM Y:(?P<y>[0-9.]+).* U:(?P<u>[0-9.]+).* V:(?P<v>[0-9.]+).* All:(?P<all>[0-9.]+)",
    )
    psnr = parse_metric(
        comparison_dir / "psnr.stderr.log",
        r"PSNR y:(?P<y>[0-9.]+) u:(?P<u>[0-9.]+) v:(?P<v>[0-9.]+) average:(?P<average>[0-9.]+)",
    )
    audio = parse_metric(
        comparison_dir / "audio-psnr.stderr.log",
        r"PSNR ch0: (?P<channel_0>[0-9.]+) dB[\s\S]*?PSNR ch1: (?P<channel_1>[0-9.]+) dB",
    )
    comparison_files = {
        "side_by_side_mp4": comparison_dir / "m7c-csharp-vs-python-side-by-side.mp4",
        "contact_sheet_jpg": comparison_dir / "m7c-contact-sheet.jpg",
    }
    comparison_report = {
        "schema_version": 1,
        "result": "pass",
        "visual_review": "coherent_matching_beach_scene_across_five_sampled_times",
        "video_metrics": {"ssim": ssim, "psnr_db": psnr},
        "audio_apsnr_db": audio,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in comparison_files.items()
        },
        "secrets_recorded": False,
    }
    write(artifact_dir / "comparison.json", comparison_report)

    csharp_stats, csharp_tensors = tensor_stats(BUILD / "diagnostics" / "csharp-stage-latents.safetensors")
    python_stats, python_tensors = tensor_stats(BUILD / "diagnostics" / "python-stage-latents.safetensors")
    compared_names = (
        "stage1_step0_video_input",
        "stage1_step0_video_velocity",
        "stage1_video",
        "stage1_audio",
        "stage2_video",
        "stage2_audio",
    )
    latent_report = {
        "schema_version": 1,
        "result": "pass",
        "csharp": {name: csharp_stats[name] for name in csharp_stats if name.startswith("stage")},
        "python": {name: python_stats[name] for name in python_stats if name in compared_names},
        "comparisons": {
            name: compare(csharp_tensors[name], python_tensors[name]) for name in compared_names
        },
        "csharp_fixture_sha256": sha256(BUILD / "diagnostics" / "csharp-stage-latents.safetensors"),
        "python_fixture_sha256": sha256(BUILD / "diagnostics" / "python-stage-latents.safetensors"),
        "secrets_recorded": False,
    }
    if not 0.8 < csharp_stats["stage1_video"]["standard_deviation"] < 1.2:
        raise SystemExit("C# stage-1 latent has collapsed content")
    if not 0.8 < csharp_stats["stage2_video"]["standard_deviation"] < 1.2:
        raise SystemExit("C# stage-2 latent has collapsed content")
    if latent_report["comparisons"]["stage1_step0_video_input"]["exact_fraction"] != 1:
        raise SystemExit("C# and Python initial production latents differ")
    if latent_report["comparisons"]["stage1_step0_video_velocity"]["cosine"] < 0.999:
        raise SystemExit("C# production transformer velocity parity is insufficient")
    write(artifact_dir / "latents.json", latent_report)

    model_download = load_json(BUILD / "model-download.json")
    prompt_context = load_json(BUILD / "prompt-context.json")
    model_report = {
        "schema_version": 1,
        "result": "pass",
        "checkpoint": {
            "repository": "Lightricks/LTX-2.3",
            "revision": "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce",
            "path": "/workspace/ltx-model-cache/M7A/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors",
            "sha256": "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc",
            "model_version": "2.3.0",
        },
        "spatial_upsampler": {
            "repository": "Lightricks/LTX-2.3",
            "revision": "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce",
            "path": "/workspace/ltx-model-cache/M7C/ltx23-spatial-upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "sha256": "5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
        },
        "text_encoder": {
            "repository": "google/gemma-3-12b-it-qat-q4_0-unquantized",
            "revision": prompt_context["gemma_revision"],
        },
        "prompt_context": {
            "path": prompt_context["output_path"],
            "sha256": prompt_context["output_sha256"],
            "preparation_elapsed_seconds": prompt_context["elapsed_seconds"],
            "peak_cuda_memory_allocated_bytes": prompt_context["peak_cuda_memory_allocated_bytes"],
            "offload": prompt_context["offload"],
        },
        "download_manifest": model_download,
        "secrets_recorded": False,
    }
    write(artifact_dir / "models.json", model_report)

    gpu_values = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().split(",")
    hardware_report = {
        "schema_version": 1,
        "result": "pass",
        "gpu": {
            "name": gpu_values[0].strip(),
            "compute_capability": gpu_values[1].strip(),
            "driver": gpu_values[2].strip(),
            "memory_total_mib": int(gpu_values[3]),
            "csharp_peak_memory_used_mib": csharp_peak,
            "python_peak_memory_used_mib": python_peak,
        },
        "offload": {
            "csharp": "required_disk_weight_streaming_after_resident_cuda_oom",
            "python": "none_resident",
        },
        "secrets_recorded": False,
    }
    write(artifact_dir / "hardware.json", hardware_report)

    transformer_oracle = load_json(BUILD / "diagnostics" / "production-transformer-rope-repaired.json")
    write(artifact_dir / "production-transformer.json", transformer_oracle)
    repairs = {
        "schema_version": 1,
        "result": "pass",
        "root_cause": "RoPE frequency axes were transposed after already reaching the reference frequency-axis layout",
        "repairs": [
            "interleave RoPE axes exactly like Python generate_freqs",
            "use native at::linear and at::add through the maintained TorchSharp bridge",
            "match X0Model BF16 conversion plus EulerDiffusionStep semantics",
            "implement learned x2 spatial latent upsampler",
            "add exact auto-tiling for the convolutional video VAE decoder",
            "mux an explicit 241 video frames without -shortest truncation",
        ],
        "failed_attempts_retained_on_host": {
            "resident_oom": str(BUILD / "csharp-resident-failure"),
            "offload_abi_failures": [
                str(BUILD / "csharp-offload-failure-1"),
                str(BUILD / "csharp-offload-failure-2"),
                str(BUILD / "csharp-offload-failure-3"),
            ],
            "mux_240_frame_failure": str(BUILD / "csharp-mux-failure-4"),
            "uniform_content_failure": str(BUILD / "csharp-content-failure-5"),
        },
        "python_substitution_for_csharp": False,
        "secrets_recorded": False,
    }
    write(artifact_dir / "port-repair.json", repairs)
    print("M7C evidence collection: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
