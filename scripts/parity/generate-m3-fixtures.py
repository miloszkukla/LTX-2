#!/usr/bin/env python3
"""Generate seeded core-model execution fixtures for M3."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


EXPECTED_TORCH = "2.13.0+cu132"
EXPECTED_CUDA = "13.2"
EXPECTED_SAFETENSORS = "0.6.2"
SEED = 20260820
REVISION = "m3-core-model-execution-v1"
FP32_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
BF16_TOLERANCE = {"rtol": 2e-2, "atol": 5e-3}


def tensor_data(tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().float().cpu().contiguous()
    return {
        "shape": list(contiguous.shape),
        "values": contiguous.flatten().tolist(),
    }


def fixture_case(
    *,
    suite: str,
    tensors: dict[str, torch.Tensor],
    expected_fp32: torch.Tensor,
    expected_bf16: torch.Tensor | None = None,
    integers: dict[str, int] | None = None,
    numbers: dict[str, float] | None = None,
    flags: dict[str, bool] | None = None,
) -> dict:
    expected = {"fp32": tensor_data(expected_fp32)}
    if expected_bf16 is not None:
        expected["bf16"] = tensor_data(expected_bf16)
    return {
        "suite": suite,
        "tensors": {name: tensor_data(value) for name, value in sorted(tensors.items())},
        "integers": integers or {},
        "numbers": numbers or {},
        "flags": flags or {},
        "expected": expected,
    }


def timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dimension: int,
    *,
    flip_sin_to_cos: bool,
    downscale_frequency_shift: float,
    scale: float,
    maximum_period: int,
) -> torch.Tensor:
    half_dimension = embedding_dimension // 2
    exponent = -math.log(maximum_period) * torch.arange(
        0,
        half_dimension,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dimension - downscale_frequency_shift)
    arguments = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    arguments = arguments * scale
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dimension:], embedding[:, :half_dimension]], dim=-1)
    if embedding_dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def feed_forward(
    input_tensor: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    hidden = input_tensor @ input_weight.T + input_bias
    hidden = F.gelu(hidden, approximate="tanh")
    return hidden @ output_weight.T + output_bias


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, query_tokens, channels = query.shape
    key_tokens = key.shape[1]
    head_dimension = channels // heads
    q = query.reshape(batch, query_tokens, heads, head_dimension).transpose(1, 2)
    k = key.reshape(batch, key_tokens, heads, head_dimension).transpose(1, 2)
    v = value.reshape(batch, key_tokens, heads, head_dimension).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dimension)
    if mask is not None:
        scores = scores + (1 - mask[:, None].to(scores)) * -10_000
    return (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous().reshape(batch, query_tokens, channels)


def patchify(input_tensor: torch.Tensor, spatial_patch: int, temporal_patch: int) -> torch.Tensor:
    batch, channels, frames, height, width = input_tensor.shape
    return (
        input_tensor.reshape(
            batch,
            channels,
            frames // temporal_patch,
            temporal_patch,
            height // spatial_patch,
            spatial_patch,
            width // spatial_patch,
            spatial_patch,
        )
        .permute(0, 1, 3, 7, 5, 2, 4, 6)
        .contiguous()
        .reshape(
            batch,
            channels * temporal_patch * spatial_patch * spatial_patch,
            frames // temporal_patch,
            height // spatial_patch,
            width // spatial_patch,
        )
    )


def unpatchify(input_tensor: torch.Tensor, spatial_patch: int, temporal_patch: int) -> torch.Tensor:
    batch, channels, frames, height, width = input_tensor.shape
    base_channels = channels // (temporal_patch * spatial_patch * spatial_patch)
    return (
        input_tensor.reshape(
            batch,
            base_channels,
            temporal_patch,
            spatial_patch,
            spatial_patch,
            frames,
            height,
            width,
        )
        .permute(0, 1, 5, 2, 6, 4, 7, 3)
        .contiguous()
        .reshape(
            batch,
            base_channels,
            frames * temporal_patch,
            height * spatial_patch,
            width * spatial_patch,
        )
    )


def pixel_norm(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor / torch.sqrt(torch.mean(input_tensor**2, dim=1, keepdim=True) + 1e-8)


def channel_statistics(
    input_tensor: torch.Tensor,
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
    *,
    normalize: bool,
    dimension: int,
) -> torch.Tensor:
    shape = [1] * input_tensor.ndim
    shape[dimension] = input_tensor.shape[dimension]
    means = means.to(input_tensor).reshape(shape)
    standard_deviations = standard_deviations.to(input_tensor).reshape(shape)
    if normalize:
        return (input_tensor - means) / standard_deviations
    return input_tensor * standard_deviations + means


def snake_beta(
    input_tensor: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    logscale: bool,
) -> torch.Tensor:
    shape = [1] * input_tensor.ndim
    shape[1] = input_tensor.shape[1]
    alpha = alpha.to(input_tensor).reshape(shape)
    beta = beta.to(input_tensor).reshape(shape)
    if logscale:
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
    return input_tensor + torch.sin(input_tensor * alpha).pow(2) / (beta + 1e-9)


def prepare_stereo_mel(input_tensor: torch.Tensor) -> torch.Tensor:
    transposed = input_tensor.transpose(2, 3).contiguous()
    return transposed.reshape(transposed.shape[0], transposed.shape[1] * transposed.shape[2], transposed.shape[3])


def build_attention_mask(
    existing_mask: torch.Tensor | None,
    noisy_tokens: int,
    new_tokens: int,
    existing_tokens: int,
    cross_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = cross_mask.shape[0]
    total = existing_tokens + new_tokens
    mask = torch.zeros((batch_size, total, total), dtype=cross_mask.dtype, device=cross_mask.device)
    if existing_mask is None:
        mask[:, :existing_tokens, :existing_tokens] = 1
    else:
        mask[:, :existing_tokens, :existing_tokens] = existing_mask
    mask[:, existing_tokens:, existing_tokens:] = 1
    mask[:, :noisy_tokens, existing_tokens:] = cross_mask.unsqueeze(1)
    mask[:, existing_tokens:, :noisy_tokens] = cross_mask.unsqueeze(2)
    return mask


def save_canonical(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, path, metadata={"fixture_revision": REVISION})
    serialized = path.read_bytes()
    header_length = struct.unpack("<Q", serialized[:8])[0]
    header = json.loads(serialized[8 : 8 + header_length])
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    canonical += b" " * (8 - len(canonical) % 8)
    path.write_bytes(struct.pack("<Q", len(canonical)) + canonical + serialized[8 + header_length :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(f"oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
    if safetensors.__version__ != EXPECTED_SAFETENSORS:
        raise SystemExit(f"safetensors mismatch: {safetensors.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("M3 fixture generation requires the accepted CUDA worker")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    device = torch.device("cuda")

    def randn(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    cases: dict[str, dict] = {}

    timesteps = torch.tensor([0.0, 1.5, 999.0], dtype=torch.float32)
    timestep_expected = timestep_embedding(
        timesteps.to(device),
        7,
        flip_sin_to_cos=True,
        downscale_frequency_shift=0,
        scale=0.75,
        maximum_period=10_000,
    )
    cases["transformer.timestep_embedding"] = fixture_case(
        suite="transformer",
        tensors={"timesteps": timesteps},
        integers={"embedding_dimension": 7, "maximum_period": 10_000},
        numbers={"downscale_frequency_shift": 0, "scale": 0.75},
        flags={"flip_sin_to_cos": True},
        expected_fp32=timestep_expected,
    )

    ff_tensors = {
        "input": randn((2, 3, 4), 0.5),
        "input_weight": randn((7, 4), 0.35),
        "input_bias": randn((7,), 0.1),
        "output_weight": randn((4, 7), 0.3),
        "output_bias": randn((4,), 0.1),
    }
    ff_fp32 = feed_forward(*(ff_tensors[name].to(device) for name in (
        "input", "input_weight", "input_bias", "output_weight", "output_bias"
    )))
    ff_bf16 = feed_forward(*(ff_tensors[name].to(device, torch.bfloat16) for name in (
        "input", "input_weight", "input_bias", "output_weight", "output_bias"
    )))
    cases["transformer.feed_forward"] = fixture_case(
        suite="transformer",
        tensors=ff_tensors,
        expected_fp32=ff_fp32,
        expected_bf16=ff_bf16,
    )

    attention_tensors = {
        "query": randn((2, 4, 8), 0.4),
        "key": randn((2, 4, 8), 0.4),
        "value": randn((2, 4, 8), 0.4),
    }
    for masked in (False, True):
        tensors = dict(attention_tensors)
        mask = None
        if masked:
            mask = torch.tensor(
                [
                    [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]],
                    [[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]],
                ],
                dtype=torch.float32,
            )
            tensors["mask"] = mask
        fp32_values = [attention_tensors[name].to(device) for name in ("query", "key", "value")]
        bf16_values = [attention_tensors[name].to(device, torch.bfloat16) for name in ("query", "key", "value")]
        fp32_output = attention(*fp32_values, heads=2, mask=None if mask is None else mask.to(device))
        bf16_output = attention(*bf16_values, heads=2, mask=None if mask is None else mask.to(device, torch.bfloat16))
        name = "transformer.masked_attention" if masked else "transformer.attention"
        cases[name] = fixture_case(
            suite="transformer",
            tensors=tensors,
            integers={"heads": 2},
            expected_fp32=fp32_output,
            expected_bf16=bf16_output,
        )

    video = randn((1, 2, 2, 4, 4), 0.6)
    patched = patchify(video.to(device), 2, 2)
    cases["vae.patchify"] = fixture_case(
        suite="vae",
        tensors={"input": video},
        integers={"spatial_patch": 2, "temporal_patch": 2},
        expected_fp32=patched,
    )
    cases["vae.unpatchify"] = fixture_case(
        suite="vae",
        tensors={"input": patched.cpu()},
        integers={"spatial_patch": 2, "temporal_patch": 2},
        expected_fp32=unpatchify(patched, 2, 2),
    )

    pixel_input = randn((1, 4, 2, 2, 2), 0.75)
    cases["vae.pixel_norm"] = fixture_case(
        suite="vae",
        tensors={"input": pixel_input},
        expected_fp32=pixel_norm(pixel_input.to(device)),
        expected_bf16=pixel_norm(pixel_input.to(device, torch.bfloat16)),
    )
    statistics_input = randn((1, 4, 2, 2, 2), 0.5)
    means = torch.tensor([-0.25, 0.1, 0.35, -0.05], dtype=torch.float32)
    standard_deviations = torch.tensor([0.5, 1.25, 0.75, 1.5], dtype=torch.float32)
    for normalize in (True, False):
        name = "vae.statistics_normalize" if normalize else "vae.statistics_denormalize"
        fp32 = channel_statistics(
            statistics_input.to(device),
            means.to(device),
            standard_deviations.to(device),
            normalize=normalize,
            dimension=1,
        )
        bf16 = channel_statistics(
            statistics_input.to(device, torch.bfloat16),
            means.to(device, torch.bfloat16),
            standard_deviations.to(device, torch.bfloat16),
            normalize=normalize,
            dimension=1,
        )
        cases[name] = fixture_case(
            suite="vae",
            tensors={"input": statistics_input, "means": means, "standard_deviations": standard_deviations},
            integers={"dimension": 1},
            expected_fp32=fp32,
            expected_bf16=bf16,
        )

    audio_input = randn((1, 3, 9), 0.8)
    alpha = torch.tensor([-0.2, 0.0, 0.25], dtype=torch.float32)
    beta = torch.tensor([0.15, -0.1, 0.3], dtype=torch.float32)
    for snake_name, snake_beta_value in (("audio.snake", alpha), ("audio.snake_beta", beta)):
        fp32 = snake_beta(
            audio_input.to(device),
            alpha.to(device),
            snake_beta_value.to(device),
            logscale=True,
        )
        bf16 = snake_beta(
            audio_input.to(device, torch.bfloat16),
            alpha.to(device, torch.bfloat16),
            snake_beta_value.to(device, torch.bfloat16),
            logscale=True,
        )
        cases[snake_name] = fixture_case(
            suite="audio_vocoder",
            tensors={"input": audio_input, "alpha": alpha, "beta": snake_beta_value},
            flags={"logscale": True},
            expected_fp32=fp32,
            expected_bf16=bf16,
        )

    stereo_mel = randn((1, 2, 3, 4), 0.4)
    cases["vocoder.prepare_stereo_mel"] = fixture_case(
        suite="audio_vocoder",
        tensors={"input": stereo_mel},
        expected_fp32=prepare_stereo_mel(stereo_mel.to(device)),
    )
    waveform = randn((1, 2, 10), 1.4)
    for use_tanh in (True, False):
        name = "vocoder.final_tanh" if use_tanh else "vocoder.final_clamp"
        activation = torch.tanh if use_tanh else lambda value: torch.clamp(value, -1, 1)
        cases[name] = fixture_case(
            suite="audio_vocoder",
            tensors={"input": waveform},
            flags={"use_tanh": use_tanh},
            expected_fp32=activation(waveform.to(device)),
            expected_bf16=activation(waveform.to(device, torch.bfloat16)),
        )

    cross_first = torch.tensor([[0.25, 0.75], [1.0, 0.5]], dtype=torch.float32)
    first_mask = build_attention_mask(None, 3, 2, 3, cross_first.to(device))
    cases["conditioning.first_mask"] = fixture_case(
        suite="conditioning",
        tensors={"cross_mask": cross_first},
        integers={"noisy_tokens": 3, "new_tokens": 2, "existing_tokens": 3},
        expected_fp32=first_mask,
    )
    cross_second = torch.tensor([[0.4], [0.9]], dtype=torch.float32)
    cases["conditioning.append_mask"] = fixture_case(
        suite="conditioning",
        tensors={"existing_mask": first_mask.cpu(), "cross_mask": cross_second},
        integers={"noisy_tokens": 3, "new_tokens": 1, "existing_tokens": 5},
        expected_fp32=build_attention_mask(first_mask, 3, 1, 5, cross_second.to(device)),
    )
    scalar_mask = torch.tensor(0.35, dtype=torch.float32)
    cases["conditioning.resolve_scalar"] = fixture_case(
        suite="conditioning",
        tensors={"attention_mask": scalar_mask},
        integers={"new_tokens": 3, "batch_size": 2},
        expected_fp32=torch.full((2, 3), 0.35, device=device),
    )
    vector_mask = torch.tensor([0.1, 0.6, 1.0], dtype=torch.float32)
    cases["conditioning.resolve_vector"] = fixture_case(
        suite="conditioning",
        tensors={"attention_mask": vector_mask},
        integers={"new_tokens": 3, "batch_size": 2},
        expected_fp32=vector_mask.to(device).unsqueeze(0).expand(2, -1),
    )

    offload_input = randn((2, 4), 0.5)
    offload_weight = randn((3, 4), 0.4)
    offload_bias = randn((3,), 0.1)
    offload_fp32 = offload_input.to(device) @ offload_weight.to(device).T + offload_bias.to(device)
    offload_bf16 = (
        offload_input.to(device, torch.bfloat16) @ offload_weight.to(device, torch.bfloat16).T
        + offload_bias.to(device, torch.bfloat16)
    )
    cases["offload.linear"] = fixture_case(
        suite="offload",
        tensors={"input": offload_input, "weight": offload_weight, "bias": offload_bias},
        expected_fp32=offload_fp32,
        expected_bf16=offload_bf16,
    )
    save_canonical(
        {"linear.weight": offload_weight, "linear.bias": offload_bias},
        output / "offload.safetensors",
    )

    manifest = {
        "schema_version": 1,
        "fixture_revision": REVISION,
        "oracle": {
            "framework": "pytorch",
            "framework_version": EXPECTED_TORCH,
            "cuda_version": EXPECTED_CUDA,
            "safetensors_version": EXPECTED_SAFETENSORS,
            "seed": SEED,
            "device": torch.cuda.get_device_name(0),
        },
        "tolerances": {"fp32": FP32_TOLERANCE, "bf16": BF16_TOLERANCE},
        "cases": cases,
        "file_sha256": {
            "offload.safetensors": hashlib.sha256((output / "offload.safetensors").read_bytes()).hexdigest(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
