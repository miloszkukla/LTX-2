#!/usr/bin/env python3
"""Collect and enforce the redacted M0 host/toolchain preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PLAN_SHA256 = "d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6"
EXPECTED_BASE_DIGEST = "sha256:293837d7ad950f195e6ba5e3c0478c50b8c80349a70ad14687f39ed8ff0a8e4a"


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def pid1_names() -> set[str]:
    try:
        return {
            item.split(b"=", 1)[0].decode()
            for item in Path("/proc/1/environ").read_bytes().split(b"\0")
            if b"=" in item
        }
    except (OSError, UnicodeDecodeError):
        return set()


def library_present(fragment: str) -> bool:
    return fragment in command("ldconfig", "-p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/preflight.json")
    args = parser.parse_args()

    gpu_lines = command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap,driver_version,temperature.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ).splitlines()
    if len(gpu_lines) != 1:
        raise SystemExit(f"exactly one GPU is required, found {len(gpu_lines)}")
    gpu_name, memory_mib, compute_capability, driver, temperature, memory_used = [
        value.strip() for value in gpu_lines[0].split(",")
    ]
    if "RTX 5090" not in gpu_name or int(memory_mib) < 32000:
        raise SystemExit("M0 requires the RTX 5090 32 GB routine worker")
    if int(driver.split(".", 1)[0]) < 580:
        raise SystemExit("CUDA 13.x minor-version compatibility requires Linux driver >=580")

    nvcc = command("nvcc", "--version")
    release = re.search(r"release\s+(\d+)\.(\d+)", nvcc)
    if not release or release.group(1) != "13":
        raise SystemExit("CUDA 13.x nvcc is required")
    sass_architecture = "sm_" + compute_capability.replace(".", "")
    dotnet = command("dotnet", "--version")
    if not dotnet.startswith("10."):
        raise SystemExit(".NET SDK 10 is required")

    disk = shutil.disk_usage(ROOT)
    memory_bytes = int(command("awk", "/MemTotal/ {print $2 * 1024}", "/proc/meminfo"))
    model_access = json.loads((ROOT / "artifacts/M0/model-access.json").read_text())
    required_disk = int(model_access["disk_gate"]["required_disk_with_20_percent_headroom_bytes"])
    if disk.total < required_disk:
        raise SystemExit(f"disk gate failed: {disk.total} < {required_disk}")
    if memory_bytes < 30 * 1024**3:
        raise SystemExit("at least 32 GB nominal host RAM is required")

    plan_sha = command("sha256sum", str(ROOT / "C_SHARP_PORT_PLAN.md")).split()[0]
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise SystemExit("C_SHARP_PORT_PLAN.md hash mismatch")

    environment_names = set(os.environ) | pid1_names()
    libraries = {
        "ffmpeg_avcodec": library_present("libavcodec.so"),
        "ffmpeg_avformat": library_present("libavformat.so"),
        "openimageio": library_present("libOpenImageIO.so"),
        "opencv_core": library_present("libopencv_core.so"),
    }
    if not all(libraries.values()):
        raise SystemExit("required ldconfig library surface is incomplete")

    vast = json.loads((ROOT / "artifacts/M0/vast-preflight.json").read_text())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result": "pass",
        "plan_sha256": plan_sha,
        "base_image": {
            "tag": "nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04",
            "linux_amd64_digest": EXPECTED_BASE_DIGEST,
        },
        "gpu": {
            "name": gpu_name,
            "memory_total_mib": int(memory_mib),
            "memory_used_mib": int(memory_used),
            "temperature_c": int(temperature),
            "compute_capability": compute_capability,
            "sass_architecture": sass_architecture,
            "driver": driver,
            "driver_policy": "CUDA 13.x minor-version compatibility; SASS-only native build; no PTX JIT",
        },
        "toolchain": {
            "cuda_release": f"{release.group(1)}.{release.group(2)}",
            "dotnet_sdk": dotnet,
            "compiler": command("cc", "--version").splitlines()[0],
            "cmake": command("cmake", "--version").splitlines()[0],
            "ninja": command("ninja", "--version"),
            "python": command("python3", "--version"),
            "uv": command("uv", "--version"),
            "ffmpeg": command("ffmpeg", "-version").splitlines()[0],
        },
        "native_libraries": libraries,
        "capacity": {
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "required_disk_with_headroom_bytes": required_disk,
            "disk_gate": "pass",
            "ram_bytes": memory_bytes,
        },
        "access": {
            "hf_token_present": "HF_TOKEN" in environment_names,
            "hf_token_value_recorded": False,
            "direct_ssh": bool(vast["current_worker"]["direct_ssh"]),
            "codex_cli_authenticated": subprocess.run(
                ["codex", "login", "status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0,
        },
    }
    if not report["access"]["hf_token_present"] or not report["access"]["codex_cli_authenticated"]:
        raise SystemExit("HF token presence and authenticated Codex CLI are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"preflight passed: {gpu_name}, {sass_architecture}, CUDA {release.group(1)}.{release.group(2)}, .NET {dotnet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
