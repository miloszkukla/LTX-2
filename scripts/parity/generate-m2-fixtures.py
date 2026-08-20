#!/usr/bin/env python3
"""Generate fixed safetensors, LoRA, and checkpoint-loader fixtures for M2."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file


EXPECTED_TORCH = "2.13.0+cu132"
EXPECTED_CUDA = "13.2"
EXPECTED_SAFETENSORS = "0.6.2"
SEED = 20260820
REVISION = "m2-storage-model-loading-v1"
FP32_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
BF16_TOLERANCE = {"rtol": 2e-2, "atol": 5e-3}


def tensor_case(tensor: torch.Tensor) -> dict:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "values": tensor.float().flatten().tolist(),
    }


def fuse(weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, strength: float) -> torch.Tensor:
    delta = torch.matmul(b.to(torch.bfloat16) * strength, a.to(torch.bfloat16)).to(torch.bfloat16)
    return (delta + weight.to(torch.bfloat16)).to(weight.dtype)


def save_canonical(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str] | None = None) -> None:
    """Use the reference writer, then canonicalize its HashMap-dependent header ordering."""
    save_file(tensors, path, metadata=metadata)
    serialized = path.read_bytes()
    header_length = struct.unpack("<Q", serialized[:8])[0]
    header = json.loads(serialized[8 : 8 + header_length])
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    canonical += b" " * (8 - len(canonical) % 8)
    payload = serialized[8 + header_length :]
    path.write_bytes(struct.pack("<Q", len(canonical)) + canonical + payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(f"oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
    if safetensors.__version__ != EXPECTED_SAFETENSORS:
        raise SystemExit(f"safetensors mismatch: {safetensors.__version__}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    metadata = {
        "config": json.dumps(
            {
                "model": {"hidden_size": 4, "layers": 1},
                "scheduler": {"steps": 8},
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "license": "LTX-2 Community License",
        "model_version": "2.5.0",
    }
    mixed = {
        "bf16_values": torch.tensor([-3.5, -0.125, 0.0, 1.5, 9.75], dtype=torch.bfloat16),
        "bool_mask": torch.tensor([True, False, True, True, False], dtype=torch.bool),
        "empty_f32": torch.empty((0, 3), dtype=torch.float32),
        "float16_values": torch.tensor([-2.25, 0.5, 4.0], dtype=torch.float16),
        "float32_matrix": torch.randn((2, 3), generator=generator, dtype=torch.float32),
        "int64_ids": torch.tensor([0, 17, -9, 2**40], dtype=torch.int64),
        "scalar_f64": torch.tensor(3.25, dtype=torch.float64),
        "uint8_pixels": torch.tensor([0, 1, 127, 255], dtype=torch.uint8),
    }
    save_canonical(mixed, output / "storage-mixed.safetensors", metadata=metadata)

    linear_weight = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    projection_weight = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    linear_bias = torch.randn((3,), generator=generator, dtype=torch.float32)
    norm_weight = torch.randn((3,), generator=generator, dtype=torch.bfloat16)
    shard_one = {
        "diffusion_model.block.linear.weight": linear_weight,
        "diffusion_model.block.proj.weight": projection_weight,
        "ignored.tensor": torch.ones((2,), dtype=torch.float32),
    }
    shard_two = {
        "diffusion_model.block.linear.bias": linear_bias,
        "diffusion_model.block.norm.weight": norm_weight,
    }
    save_canonical(shard_one, output / "checkpoint-00001-of-00002.safetensors", metadata=metadata)
    save_canonical(shard_two, output / "checkpoint-00002-of-00002.safetensors")

    linear_a = torch.randn((2, 4), generator=generator, dtype=torch.float32)
    linear_b = torch.randn((3, 2), generator=generator, dtype=torch.float32)
    projection_a = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    projection_b = torch.randn((2, 2), generator=generator, dtype=torch.bfloat16)
    lora = {
        "diffusion_model.block.linear.lora_A.weight": linear_a,
        "diffusion_model.block.linear.lora_B.weight": linear_b,
        "diffusion_model.block.proj.lora_A.weight": projection_a,
        "diffusion_model.block.proj.lora_B.weight": projection_b,
    }
    lora_metadata = {
        "adapter": "m2-fixed-lora",
        "format": "pt",
        "reference_downscale_factor": "2",
    }
    save_canonical(lora, output / "lora.safetensors", metadata=lora_metadata)

    strength = 0.625
    fused_linear = fuse(linear_weight, linear_a, linear_b, strength)
    fused_projection = fuse(projection_weight, projection_a, projection_b, strength)
    unfused_linear = fuse(fused_linear, linear_a, linear_b, -strength)
    unfused_projection = fuse(fused_projection, projection_a, projection_b, -strength)

    manifest = {
        "schema_version": 1,
        "fixture_revision": REVISION,
        "oracle": {
            "framework": "pytorch",
            "framework_version": EXPECTED_TORCH,
            "cuda_version": EXPECTED_CUDA,
            "safetensors_version": EXPECTED_SAFETENSORS,
            "seed": SEED,
        },
        "tolerances": {"fp32": FP32_TOLERANCE, "bf16": BF16_TOLERANCE},
        "metadata": metadata,
        "mixed_tensors": {key: tensor_case(value) for key, value in sorted(mixed.items())},
        "checkpoint": {
            "paths": ["checkpoint-00001-of-00002.safetensors", "checkpoint-00002-of-00002.safetensors"],
            "expected": {
                "block.linear.bias": tensor_case(linear_bias),
                "block.linear.weight": tensor_case(linear_weight),
                "block.norm.weight": tensor_case(norm_weight),
                "block.proj.weight": tensor_case(projection_weight),
            },
        },
        "lora": {
            "path": "lora.safetensors",
            "metadata": lora_metadata,
            "strength": strength,
            "mapped_keys": sorted(key.removeprefix("diffusion_model.") for key in lora),
            "expected_fused": {
                "block.linear.weight": tensor_case(fused_linear),
                "block.proj.weight": tensor_case(fused_projection),
            },
            "expected_unfused": {
                "block.linear.weight": tensor_case(unfused_linear),
                "block.proj.weight": tensor_case(unfused_projection),
            },
        },
    }
    fixture_files = [
        "checkpoint-00001-of-00002.safetensors",
        "checkpoint-00002-of-00002.safetensors",
        "lora.safetensors",
        "storage-mixed.safetensors",
    ]
    manifest["file_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in fixture_files
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
