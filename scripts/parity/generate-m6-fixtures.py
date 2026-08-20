#!/usr/bin/env python3
"""Generate deterministic preprocessing and one-step low-VRAM LoRA fixtures for M6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


EXPECTED_TORCH = "2.13.0+cu132"
EXPECTED_CUDA = "13.2"
SEED = 20260820
REVISION = "m6-low-vram-training-v1"
FP32_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
REDUCED_TOLERANCE = {"rtol": 2e-2, "atol": 5e-3}


def values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().flatten().tolist()


def tensor_case(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": list(tensor.shape), "values": values(tensor)}


def quantize_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = tensor.detach().float().cpu()
    maximum = source.abs().amax(dim=1)
    scales = torch.where(maximum > 0, maximum / 127.0, torch.zeros_like(maximum))
    normalized = torch.where(scales[:, None] > 0, source / scales[:, None], torch.zeros_like(source))
    quantized = torch.round(normalized).clamp(-127, 127).to(torch.int8)
    dequantized = quantized.float() * scales[:, None]
    return quantized, scales, dequantized


def optimizer_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    first = (1.0 - beta1) * gradient.float().cpu()
    second = (1.0 - beta2) * gradient.float().cpu().square()
    first_q, first_scale, first_dequantized = quantize_rows(first)
    second_q, second_scale, second_dequantized = quantize_rows(second)
    first_hat = first_dequantized / (1.0 - beta1)
    second_hat = second_dequantized / (1.0 - beta2)
    updated = parameter.float().cpu() * (1.0 - learning_rate * weight_decay)
    updated -= learning_rate * first_hat / (second_hat.sqrt() + epsilon)
    state = {
        "first_values": first_q.flatten().tolist(),
        "first_scales": values(first_scale),
        "second_values": second_q.flatten().tolist(),
        "second_scales": values(second_scale),
    }
    return updated, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(f"oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("M6 CUDA oracle is unavailable")

    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    video = torch.randn((3, 2, 2, 2), generator=generator, dtype=torch.float32)
    video_weight = torch.randn((4, 3), generator=generator, dtype=torch.float32) * 0.35
    video_bias = torch.randn((4,), generator=generator, dtype=torch.float32) * 0.1
    caption_features = torch.randn((3, 5), generator=generator, dtype=torch.float32)
    connector_weight = torch.randn((4, 5), generator=generator, dtype=torch.float32) * 0.25
    connector_bias = torch.randn((4,), generator=generator, dtype=torch.float32) * 0.05
    attention_mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

    pixels = video.permute(1, 2, 3, 0).reshape(-1, video.shape[0])
    encoded = (pixels @ video_weight.T + video_bias).reshape(2, 2, 2, 4).permute(3, 0, 1, 2).contiguous()
    prompt_embeds = caption_features @ connector_weight.T + connector_bias

    base_weight = torch.randn((4, 4), generator=generator, dtype=torch.float32) * 0.3
    base_q, base_scales, base_dequantized = quantize_rows(base_weight)
    lora_a_initial = torch.randn((2, 4), generator=generator, dtype=torch.float32) * 0.08
    lora_b_initial = torch.randn((4, 2), generator=generator, dtype=torch.float32) * 0.06
    noise = torch.randn((1, 8, 4), generator=generator, dtype=torch.float32)

    options = {
        "rank": 2,
        "alpha": 2.0,
        "sigma": 0.37,
        "learning_rate": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "max_gradient_norm": 1.0,
        "activation_chunk_size": 3,
        "mixed_precision": "bf16",
        "base_quantization": "int8-rowwise",
        "optimizer": "cpu-offloaded-adamw8bit",
        "gradient_checkpointing": True,
        "batch_size": 1,
    }

    device = torch.device("cuda")
    latents = encoded.permute(1, 2, 3, 0).reshape(1, 8, 4).to(device)
    mask = attention_mask.to(device)
    embeds = prompt_embeds.to(device)
    context = (embeds * mask[:, None]).sum(dim=0) / mask.sum().clamp(min=1.0)
    noise_cuda = noise.to(device)
    sigma = options["sigma"]
    noisy = (1.0 - sigma) * latents + sigma * noise_cuda
    targets = noise_cuda - latents
    model_input = (noisy + context.reshape(1, 1, -1)).to(torch.bfloat16)
    base_cuda = base_dequantized.to(device=device, dtype=torch.bfloat16)
    lora_a = lora_a_initial.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    lora_b = lora_b_initial.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    scale = options["alpha"] / options["rank"]

    weighted_losses: list[torch.Tensor] = []
    token_count = model_input.shape[1]
    for start in range(0, token_count, options["activation_chunk_size"]):
        length = min(options["activation_chunk_size"], token_count - start)
        effective = base_cuda + (lora_b @ lora_a) * scale
        prediction = model_input[:, start : start + length] @ effective.T
        chunk_target = targets[:, start : start + length].float()
        chunk_loss = (prediction.float() - chunk_target).square().mean()
        weight = length / token_count
        (chunk_loss * weight).backward()
        weighted_losses.append(chunk_loss.detach() * weight)

    loss = torch.stack(weighted_losses).sum()
    raw_a_gradient = lora_a.grad.detach().float().cpu()
    raw_b_gradient = lora_b.grad.detach().float().cpu()
    global_norm = math.sqrt(float(raw_a_gradient.square().sum() + raw_b_gradient.square().sum()))
    clip_scale = min(1.0, options["max_gradient_norm"] / (global_norm + 1e-6))
    clipped_a_gradient = raw_a_gradient * clip_scale
    clipped_b_gradient = raw_b_gradient * clip_scale
    a_before = lora_a.detach().float().cpu()
    b_before = lora_b.detach().float().cpu()
    updated_a, a_state = optimizer_step(
        a_before,
        clipped_a_gradient,
        options["learning_rate"],
        options["beta1"],
        options["beta2"],
        options["epsilon"],
        options["weight_decay"],
    )
    updated_b, b_state = optimizer_step(
        b_before,
        clipped_b_gradient,
        options["learning_rate"],
        options["beta1"],
        options["beta2"],
        options["epsilon"],
        options["weight_decay"],
    )

    manifest = {
        "schema_version": 1,
        "fixture_revision": REVISION,
        "oracle": {
            "framework": "pytorch",
            "framework_version": EXPECTED_TORCH,
            "cuda_version": EXPECTED_CUDA,
            "seed": SEED,
        },
        "tolerances": {"fp32": FP32_TOLERANCE, "bf16": REDUCED_TOLERANCE},
        "preprocessing": {
            "sample_id": "sample-0001",
            "fps": 24.0,
            "inputs": {
                "video": tensor_case(video),
                "video_encoder_weight": tensor_case(video_weight),
                "video_encoder_bias": tensor_case(video_bias),
                "caption_features": tensor_case(caption_features),
                "connector_weight": tensor_case(connector_weight),
                "connector_bias": tensor_case(connector_bias),
                "prompt_attention_mask": tensor_case(attention_mask),
            },
            "expected": {
                "video_latents": tensor_case(encoded),
                "video_prompt_embeds": tensor_case(prompt_embeds),
                "prompt_attention_mask": tensor_case(attention_mask),
            },
        },
        "training": {
            "options": options,
            "base_weight": tensor_case(base_weight),
            "base_quantized": {
                "rows": 4,
                "columns": 4,
                "values": base_q.flatten().tolist(),
                "scales": values(base_scales),
                "dequantized": values(base_dequantized),
            },
            "lora_a": tensor_case(lora_a_initial),
            "lora_b": tensor_case(lora_b_initial),
            "noise": tensor_case(noise),
            "expected": {
                "noisy_latents": tensor_case(noisy),
                "velocity_targets": tensor_case(targets),
                "loss": float(loss.cpu()),
                "lora_a_gradient": tensor_case(raw_a_gradient),
                "lora_b_gradient": tensor_case(raw_b_gradient),
                "gradient_global_norm": global_norm,
                "gradient_clip_scale": clip_scale,
                "updated_lora_a": tensor_case(updated_a),
                "updated_lora_b": tensor_case(updated_b),
                "lora_a_optimizer_state": a_state,
                "lora_b_optimizer_state": b_state,
            },
        },
    }
    serialized = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    (args.output / "manifest.json").write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    print(json.dumps({"fixture_revision": REVISION, "manifest_sha256": digest, "result": "pass"}, sort_keys=True))


if __name__ == "__main__":
    main()
