#!/usr/bin/env python3
"""Generate the seeded M1 foundation fixture with the pinned Python oracle."""

import argparse
import json
from pathlib import Path

import torch


EXPECTED_TORCH = "2.13.0+cu132"
EXPECTED_CUDA = "13.2"
SEED = 20260820


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(
            f"oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}"
        )

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    input_tensor = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    weight = torch.randn((3, 2), generator=generator, dtype=torch.float32)
    bias = torch.randn((2,), generator=generator, dtype=torch.float32)
    expected = torch.nn.functional.silu(input_tensor @ weight + bias)

    affine_input = torch.randn((8,), generator=generator, dtype=torch.float32)
    affine_scale = 1.75
    affine_bias = -0.125
    affine_expected = affine_input * affine_scale + affine_bias

    fixture = {
        "schema_version": 1,
        "fixture_revision": "m1-foundation-v1",
        "oracle": {
            "framework": "pytorch",
            "framework_version": EXPECTED_TORCH,
            "cuda_version": EXPECTED_CUDA,
            "seed": SEED,
        },
        "tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "torchsharp_case": {
            "name": "matmul_bias_silu_fp32",
            "input": input_tensor.flatten().tolist(),
            "input_shape": list(input_tensor.shape),
            "weight": weight.flatten().tolist(),
            "weight_shape": list(weight.shape),
            "bias": bias.tolist(),
            "expected": expected.flatten().tolist(),
            "expected_shape": list(expected.shape),
        },
        "native_case": {
            "name": "affine_f32",
            "input": affine_input.tolist(),
            "scale": affine_scale,
            "bias": affine_bias,
            "expected": affine_expected.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
