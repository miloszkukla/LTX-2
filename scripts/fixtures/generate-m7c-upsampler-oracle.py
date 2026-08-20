#!/usr/bin/env python3
"""Generate a fixed learned-spatial-upscaler oracle for the M7C C# repair."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.upsampler import LatentUpsamplerConfigurator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upsampler", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    values = torch.linspace(-2, 2, 1 * 128 * 2 * 2 * 3, dtype=torch.float32)
    input_latent = values.reshape(1, 128, 2, 2, 3).to(device=device, dtype=dtype)
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        means = checkpoint.get_tensor("vae.per_channel_statistics.mean-of-means").to(device=device, dtype=dtype)
        deviations = checkpoint.get_tensor("vae.per_channel_statistics.std-of-means").to(
            device=device, dtype=dtype
        )
    means = means.reshape(1, 128, 1, 1, 1)
    deviations = deviations.reshape(1, 128, 1, 1, 1)
    model = Builder(
        model_path=args.upsampler,
        model_class_configurator=LatentUpsamplerConfigurator,
    ).build(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        output = model(input_latent * deviations + means)
        output = (output - means) / deviations

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"input": input_latent.cpu().contiguous(), "expected": output.cpu().contiguous()},
        str(destination),
        metadata={
            "fixture_revision": "m7c-learned-spatial-upscaler-v1",
            "dtype": "bfloat16",
            "rtol": "0.02",
            "atol": "0.005",
        },
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
