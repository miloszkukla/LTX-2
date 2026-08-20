#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path


SEED = 20260820
WIDTH = 64
HEIGHT = 64
FRAMES = 9
FPS = 8
SAMPLE_RATE = 16000
MODES = [
    ("ltx_pipelines.a2vid_two_stage", "a2vid-two-stage", "A2VidPipelineTwoStage"),
    ("ltx_pipelines.dfr_pipeline", "dfr-pipeline", "DFRPipeline"),
    ("ltx_pipelines.distilled", "distilled", "DistilledPipeline"),
    ("ltx_pipelines.dubit", "dubit", "DubItPipeline"),
    ("ltx_pipelines.hdr_ic_lora", "hdr-ic-lora", "HDRICLoraPipeline"),
    ("ltx_pipelines.ic_lora", "ic-lora", "ICLoraPipeline"),
    ("ltx_pipelines.keyframe_interpolation", "keyframe-interpolation", "KeyframeInterpolationPipeline"),
    ("ltx_pipelines.retake", "retake", "RetakePipeline"),
    ("ltx_pipelines.t2a_one_stage", "t2a-one-stage", "T2AOneStagePipeline"),
    ("ltx_pipelines.ti2vid_one_stage", "ti2vid-one-stage", "TI2VidOneStagePipeline"),
    ("ltx_pipelines.ti2vid_two_stages", "ti2vid-two-stages", "TI2VidTwoStagesPipeline"),
    ("ltx_pipelines.ti2vid_two_stages_hq", "ti2vid-two-stages-hq", "TI2VidTwoStagesHQPipeline"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeded_video(mode_index: int) -> bytes:
    state = (SEED ^ 0x9E3779B9 ^ ((mode_index + 1) * 0x85EBCA6B)) & 0xFFFFFFFF
    output = bytearray(WIDTH * HEIGHT * FRAMES * 3)
    for index in range(len(output)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        output[index] = state & 0xFF
    return bytes(output)


def audio_samples(mode_index: int) -> list[float]:
    count = round(FRAMES / FPS * SAMPLE_RATE)
    frequency = 220 + mode_index * 17
    phase = (SEED % 97) / 97.0
    return [0.2 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE + phase) for index in range(count)]


def write_wave(path: Path, samples: list[float]) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(
            b"".join(struct.pack("<h", max(-32768, min(32767, round(sample * 32767)))) for sample in samples)
        )


def run(command: list[str], *, input_bytes: bytes | None = None) -> None:
    subprocess.run(command, input=input_bytes, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate-m5-fixtures.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    if len(output.parts) < 3:
        raise SystemExit("refusing an unsafe fixture output directory")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    input_audio = audio_samples(0)
    write_wave(output / "input.wav", input_audio)
    input_pixels = bytes((index * 29 + (index // 3) * 7 + 13) & 0xFF for index in range(WIDTH * HEIGHT * FRAMES * 3))
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(FPS), "-i", "pipe:0", "-i", str(output / "input.wav"),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest",
            "-map_metadata", "-1", "-fflags", "+bitexact", str(output / "input.mp4"),
        ],
        input_bytes=input_pixels,
    )
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-i", "pipe:0", "-frames:v", "1", "-map_metadata", "-1",
            str(output / "input.png"),
        ],
        input_bytes=input_pixels[: WIDTH * HEIGHT * 3],
    )

    exr_directory = output / "input_exr"
    exr_directory.mkdir()
    for frame in range(FRAMES):
        colors = (0.25 + frame / 32, 0.5 + frame / 16, 2.0 + frame / 8)
        run(
            [
                "oiiotool", "--create", f"{WIDTH}x{HEIGHT}", "3",
                f"--fill:color={colors[0]},{colors[1]},{colors[2]}", f"{WIDTH}x{HEIGHT}",
                "--attrib", "DateTime", "2026:08:20 00:00:00",
                "--attrib", "oiio:ColorSpace", "scene_linear", "--nosoftwareattrib",
                "-o", str(exr_directory / f"frame_{frame:05d}.exr"),
            ]
        )
    shutil.copyfile(exr_directory / "frame_00000.exr", output / "input.exr")

    fixture_paths = sorted(path for path in output.rglob("*") if path.is_file())
    fixture_sha = {str(path.relative_to(output)): sha256(path) for path in fixture_paths}
    manifest = {
        "schema_version": 1,
        "fixture_revision": "m5-media-pipelines-v1",
        "seed": SEED,
        "shape": {"width": WIDTH, "height": HEIGHT, "frames": FRAMES, "fps": FPS},
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "sample_count": len(input_audio),
            "first_samples": [round(sample * 32767) / 32768 for sample in input_audio[:8]],
        },
        "oracle": {
            "implementation": "python-standard-library-xorshift32",
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "openimageio": subprocess.check_output(["oiiotool", "--version"], text=True).strip(),
        },
        "tolerances": {"fp32": {"rtol": 1e-4, "atol": 1e-5}},
        "pipeline_modes": [
            {
                "module": module,
                "command": command,
                "pipeline": pipeline,
                "preview_sha256": hashlib.sha256(seeded_video(index)).hexdigest(),
                "audio_first_samples": audio_samples(index)[:8],
            }
            for index, (module, command, pipeline) in enumerate(MODES)
        ],
        "fixture_sha256": fixture_sha,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
