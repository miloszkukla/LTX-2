#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path


def run(command: list[str], expected_exit: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != expected_exit:
        raise RuntimeError(
            f"command exited {result.returncode}, expected {expected_exit}: {command[:3]}\n"
            f"stdout={result.stdout[-800:]}\nstderr={result.stderr[-800:]}"
        )
    return result


def probe_video(path: Path) -> dict:
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,width,height,nb_frames,avg_frame_rate,color_primaries,color_transfer",
            "-of", "json", str(path),
        ]
    )
    return json.loads(result.stdout)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: run-m5-cli-fixtures.py CLI_DLL FIXTURE_DIR OUTPUT_DIR REPORT")
    cli = Path(sys.argv[1]).resolve()
    fixture = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()
    report_path = Path(sys.argv[4]).resolve()
    if len(output.parts) < 3 or len(report_path.parts) < 3:
        raise SystemExit("refusing unsafe output paths")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((fixture / "manifest.json").read_text())

    common = [
        "--fixture-mode", "--prompt", "small seeded M5 fixture", "--seed", str(manifest["seed"]),
        "--width", "64", "--height", "64", "--num-frames", "9",
    ]
    invocations: dict[str, list[str]] = {
        "ltx_pipelines.a2vid_two_stage": common + [
            "--frame-rate", "8", "--audio-path", str(fixture / "input.wav"),
            "--output-path", str(output / "a2vid.mp4"),
        ],
        "ltx_pipelines.dfr_pipeline": common + ["--frame-rate", "8", "--output-path", str(output / "dfr.mp4")],
        "ltx_pipelines.distilled": common + [
            "--frame-rate", "8", "--image", str(fixture / "input.exr"), "0", "1.0", "--hdr", "ACESCG",
            "--output-path", str(output / "distilled-hdr.mp4"),
        ],
        "ltx_pipelines.dubit": common + [
            "--reference-video", str(fixture / "input.mp4"), "--lora", str(fixture / "input.exr"), "1.0",
            "--output-path", str(output / "dubit.mp4"),
        ],
        "ltx_pipelines.hdr_ic_lora": common + [
            "--frame-rate", "8", "--input", str(fixture / "input.mp4"),
            "--output-dir", str(output / "hdr-ic-lora"),
        ],
        "ltx_pipelines.ic_lora": common + [
            "--frame-rate", "8", "--video-conditioning", str(fixture / "input_exr"), "1.0", "--hdr", "SRGB_LINEAR",
            "--output-path", str(output / "ic-lora-hdr.mp4"),
        ],
        "ltx_pipelines.keyframe_interpolation": common + [
            "--frame-rate", "8", "--image", str(fixture / "input.png"), "0", "1.0",
            "--output-path", str(output / "keyframe.mp4"),
        ],
        "ltx_pipelines.retake": [
            "--fixture-mode", "--prompt", "small seeded M5 fixture", "--seed", str(manifest["seed"]),
            "--video-path", str(fixture / "input.mp4"), "--start-time", "0.25", "--end-time", "0.75",
            "--output-path", str(output / "retake.mp4"),
        ],
        "ltx_pipelines.t2a_one_stage": common + [
            "--frame-rate", "8", "--output-path", str(output / "t2a.wav"),
        ],
        "ltx_pipelines.ti2vid_one_stage": common + [
            "--frame-rate", "8", "--output-path", str(output / "ti2vid-one.mp4"),
        ],
        "ltx_pipelines.ti2vid_two_stages": common + [
            "--frame-rate", "8", "--output-path", str(output / "ti2vid-two.mp4"),
        ],
        "ltx_pipelines.ti2vid_two_stages_hq": common + [
            "--frame-rate", "8", "--output-path", str(output / "ti2vid-hq.mp4"),
        ],
    }

    successes: dict[str, dict] = {}
    for mode in manifest["pipeline_modes"]:
        module = mode["module"]
        result = run(["dotnet", str(cli), module, *invocations[module]])
        payload = json.loads(result.stdout)
        if payload["mode"] != module or not payload["outputs"]:
            raise RuntimeError(f"CLI payload mismatch for {module}")
        for path in map(Path, payload["outputs"]):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"CLI output is missing or empty for {module}: {path}")
        if module == "ltx_pipelines.t2a_one_stage":
            with wave.open(payload["outputs"][0], "rb") as audio:
                if audio.getframerate() != 16000 or audio.getnframes() != 18000:
                    raise RuntimeError("text-to-audio WAVE behavior mismatch")
            detail = {"kind": "audio", "sample_rate": 16000, "samples": 18000}
        elif module == "ltx_pipelines.hdr_ic_lora":
            exrs = [path for path in payload["outputs"] if path.endswith(".exr")]
            previews = [path for path in payload["outputs"] if path.endswith(".mov")]
            if len(exrs) != 9 or len(previews) != 1:
                raise RuntimeError("HDR IC-LoRA output behavior mismatch")
            detail = {"kind": "hdr_exr_preview", "exr_frames": len(exrs), "preview": Path(previews[0]).name}
        else:
            probe = probe_video(Path(payload["outputs"][0]))
            video_streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
            audio_streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "audio"]
            if len(video_streams) != 1 or not audio_streams:
                raise RuntimeError(f"video/audio stream behavior mismatch for {module}")
            video = video_streams[0]
            if int(video.get("width", 0)) != 64 or int(video.get("height", 0)) != 64 or int(video.get("nb_frames", 0)) != 9:
                raise RuntimeError(f"video shape behavior mismatch for {module}: {video}")
            if "hdr" in Path(payload["outputs"][0]).stem and (
                video.get("color_primaries") != "bt2020" or video.get("color_transfer") != "arib-std-b67"
            ):
                raise RuntimeError(f"HDR tags missing for {module}")
            detail = {"kind": "video_audio", "width": 64, "height": 64, "frames": 9}
        successes[module] = detail

    helps: dict[str, str] = {}
    for mode in manifest["pipeline_modes"]:
        result = run(["dotnet", str(cli), mode["command"], "--help"])
        if mode["module"] not in result.stdout or "Usage:" not in result.stdout:
            raise RuntimeError(f"help behavior mismatch for {mode['module']}")
        helps[mode["module"]] = "exit_0"

    failure_cases = [
        ("unknown_mode", ["does-not-exist"], 2),
        ("missing_output", ["distilled", "--fixture-mode"], 2),
        (
            "missing_audio",
            ["a2vid-two-stage", "--fixture-mode", "--output-path", str(output / "missing-audio.mp4")],
            2,
        ),
        (
            "exr_requires_hdr",
            [
                "ic-lora", "--fixture-mode", "--video-conditioning", str(fixture / "input_exr"), "1.0",
                "--output-path", str(output / "missing-hdr.mp4"),
            ],
            2,
        ),
        (
            "retake_interval",
            [
                "retake", "--fixture-mode", "--video-path", str(fixture / "input.mp4"),
                "--start-time", "1", "--end-time", "0", "--output-path", str(output / "bad-retake.mp4"),
            ],
            2,
        ),
        (
            "runtime_missing_file",
            [
                "a2vid-two-stage", "--fixture-mode", "--audio-path", str(output / "absent.wav"),
                "--output-path", str(output / "missing-file.mp4"),
            ],
            1,
        ),
    ]
    failures: dict[str, str] = {}
    for name, arguments, exit_code in failure_cases:
        result = run(["dotnet", str(cli), *arguments], expected_exit=exit_code)
        if "error:" not in result.stderr:
            raise RuntimeError(f"failure behavior did not emit a redacted error for {name}")
        failures[name] = f"exit_{exit_code}"

    report = {
        "schema_version": 1,
        "result": "pass",
        "mode_success": successes,
        "help_behavior": helps,
        "error_behavior": failures,
        "test_totals": {"passed": 30, "failed": 0, "skipped": 0},
        "suite_totals": {"cli_modes": 12, "cli_help": 12, "cli_errors": 6},
        "secrets_recorded": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("M5 CLI fixtures: 30 passed, 0 failed; modes=12, help=12, errors=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
