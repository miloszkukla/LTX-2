#!/usr/bin/env python3
"""Generate deterministic CUDA, quantization, and LoRA fixtures for M4."""

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
REVISION = "m4-cuda-quantization-lora-v1"
FP32_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
BF16_FP8_TOLERANCE = {"rtol": 2e-2, "atol": 5e-3}

KERNELS = [
    "fused_add_round_kernel",
    "_na3d_kernel",
    "_fused_up_mul_kernel",
    "_fused_gate_up_swiglu_kernel",
    "sm90_fp8_gemm_1d2d_impl",
    "sm90_fp8_gemm_1d2d_bias_impl",
    "gemm_fp8_kernel",
    "quantize_kernel",
    "quantize_tiled_kernel",
    "dequantize_kernel",
    "mul_scalars_kernel",
    "amax_scale_kernel",
    "fp6_pack_kernel",
    "fp6_unpack_kernel",
    "norm_rope_cvt_kernel",
    "_rms_norm_split_rope_kernel",
    "_quantize",
    "_kernel",
    "_block_quant_norm_kernel",
    "_gelu",
    "_block_quant_kernel",
    "_quant_rms_sum_mult_kernel",
    "_gated_attention_kernel",
    "_blockwise_dequantize_kernel",
]

BINDINGS = [
    "fp8_gemm_nt_sm90",
    "fp8_gemm_nt_sm89",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "scaled_mm_nvfp4",
    "mul_scalars",
    "amax_scale",
    "gemm_alpha_mode",
    "set_gemm_alpha_on_device",
    "set_gemm_autotune",
    "probe_gemm_support",
    "rms_norm_rope",
    "fp6_pack",
    "fp6_unpack",
    "rms_norm_split_rope",
]


def values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().flatten().tolist()


def fp8_encode(tensor: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    encoded = tensor.float().to(torch.float8_e4m3fn)
    return encoded.view(torch.uint8).flatten().tolist(), encoded.float()


def e2m1_encode(value: float) -> int:
    sign = 8 if value < 0 else 0
    magnitude = abs(value)
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    index = min(range(len(levels)), key=lambda candidate: (abs(levels[candidate] - magnitude), candidate & 1))
    return sign | index


def e2m1_decode(nibble: int) -> float:
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    value = levels[nibble & 7]
    return -value if nibble & 8 else value


def swizzled_offset(row: int, column: int, padded_columns: int) -> int:
    tile = (row // 128) * (padded_columns // 4) + column // 4
    local_row = row % 128
    return tile * 512 + (local_row % 32) * 16 + (local_row // 32) * 4 + column % 4


def quantize_nvfp4(tensor: torch.Tensor, high_first: bool = True) -> dict:
    rows, columns = tensor.shape
    per_tensor_scale = float(tensor.abs().nan_to_num().amax()) / 2688.0
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = (((columns // 16) + 3) // 4) * 4
    packed = bytearray(rows * columns // 2)
    block_scales = bytearray(padded_rows * padded_columns)
    decoded = torch.empty_like(tensor, dtype=torch.float32)
    for row in range(rows):
        for block_column in range(columns // 16):
            block = tensor[row, block_column * 16 : (block_column + 1) * 16].float()
            unquantized_scale = min(float(block.abs().amax()) / 6.0 / per_tensor_scale, 448.0) if per_tensor_scale else 0.0
            encoded_scale_tensor = torch.tensor([unquantized_scale], dtype=torch.float32).to(torch.float8_e4m3fn)
            encoded_scale = int(encoded_scale_tensor.view(torch.uint8)[0])
            scale = float(encoded_scale_tensor.float()[0]) * per_tensor_scale
            block_scales[swizzled_offset(row, block_column, padded_columns)] = encoded_scale
            for pair in range(8):
                first = e2m1_encode(float(block[pair * 2]) / scale) if scale else 0
                second = e2m1_encode(float(block[pair * 2 + 1]) / scale) if scale else 0
                byte = (first << 4 | second) if high_first else (second << 4 | first)
                packed[row * (columns // 2) + block_column * 8 + pair] = byte
                decoded[row, block_column * 16 + pair * 2] = e2m1_decode(first) * scale
                decoded[row, block_column * 16 + pair * 2 + 1] = e2m1_decode(second) * scale
    return {
        "rows": rows,
        "columns": columns,
        "per_tensor_scale": per_tensor_scale,
        "high_nibble_first": high_first,
        "packed": list(packed),
        "block_scales": list(block_scales),
        "dequantized": values(decoded),
    }


def window_bounds(length: int, kernel: int, causal: bool) -> tuple[list[int], list[int]]:
    starts = []
    ends = []
    if causal:
        for index in range(length):
            starts.append(max(0, index - kernel + 1))
            ends.append(index + 1)
    else:
        kernel = min(kernel, length)
        for index in range(length):
            start = min(max(index - kernel // 2, 0), length - kernel)
            starts.append(start)
            ends.append(start + kernel)
    return starts, ends


def na3d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kernels: tuple[int, int, int]) -> torch.Tensor:
    batch, time, height, width, heads, head_dim = query.shape
    bounds = [window_bounds(length, kernel, False) for length, kernel in zip((time, height, width), kernels, strict=True)]
    output = torch.empty_like(query)
    scale = head_dim**-0.5
    for item in range(batch):
        for qt in range(time):
            for qh in range(height):
                for qw in range(width):
                    keys = key[
                        item,
                        bounds[0][0][qt] : bounds[0][1][qt],
                        bounds[1][0][qh] : bounds[1][1][qh],
                        bounds[2][0][qw] : bounds[2][1][qw],
                    ].reshape(-1, heads, head_dim)
                    vals = value[
                        item,
                        bounds[0][0][qt] : bounds[0][1][qt],
                        bounds[1][0][qh] : bounds[1][1][qh],
                        bounds[2][0][qw] : bounds[2][1][qw],
                    ].reshape(-1, heads, head_dim)
                    for head in range(heads):
                        scores = (keys[:, head] * query[item, qt, qh, qw, head]).sum(-1) * scale
                        output[item, qt, qh, qw, head] = torch.softmax(scores, dim=0) @ vals[:, head]
    return output


def pack_fp6(source: list[int]) -> tuple[list[int], list[int]]:
    six = [((item >> 7) << 5) | (((item >> 4) & 1) << 4) | (item & 15) for item in source]
    packed: list[int] = []
    unpacked: list[int] = []
    for offset in range(0, len(six), 4):
        a, b, c, d = six[offset : offset + 4]
        packed.extend((((a << 2) | (b >> 4)) & 255, ((b << 4) | (c >> 2)) & 255, ((c << 6) | d) & 255))
        unpacked.extend((((a >> 5) << 7) | (((a >> 4) & 1) << 4) | (a & 15),
                         ((b >> 5) << 7) | (((b >> 4) & 1) << 4) | (b & 15),
                         ((c >> 5) << 7) | (((c >> 4) & 1) << 4) | (c & 15),
                         ((d >> 5) << 7) | (((d >> 4) & 1) << 4) | (d & 15)))
    return packed, unpacked


def block_quantize(tensor: torch.Tensor, block_size: int, use_gelu: bool = False) -> dict:
    transformed = tensor * torch.sigmoid(1.702 * tensor) if use_gelu else tensor
    rows, columns = transformed.shape
    encoded: list[int] = []
    scales: list[float] = []
    decoded: list[float] = []
    for row in range(rows):
        for block_column in range(columns // block_size):
            block = transformed[row, block_column * block_size : (block_column + 1) * block_size]
            scale = float(block.abs().amax()) / 448.0
            code, reconstructed = fp8_encode(block / scale if scale else torch.zeros_like(block))
            encoded.extend(code)
            scales.append(scale)
            decoded.extend(values(reconstructed * scale))
    return {"values": encoded, "scales": scales, "dequantized": decoded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(f"oracle ABI mismatch: torch={torch.__version__}, cuda={torch.version.cuda}")
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    fused_delta = torch.tensor([0.25, -0.5, 1.0, -2.0, 4.0, -8.0], dtype=torch.float32)
    fused_weight = torch.tensor([0.5, 1.0, -2.0, 4.0, -8.0, 16.0], dtype=torch.float32)

    na_shape = (1, 2, 2, 3, 2, 4)
    query = torch.randn(na_shape, generator=generator)
    key = torch.randn(na_shape, generator=generator)
    value = torch.randn(na_shape, generator=generator)
    na_expected = na3d(query, key, value, (2, 2, 3))

    swiglu_input = torch.randn((3, 4), generator=generator)
    gate_weight = torch.randn((5, 4), generator=generator)
    up_weight = torch.randn((5, 4), generator=generator)
    gate = torch.randn((3, 5), generator=generator)
    gate_linear = swiglu_input @ gate_weight.T
    up_linear = swiglu_input @ up_weight.T

    gemm_left_source = torch.randn((3, 8), generator=generator)
    gemm_right_source = torch.randn((5, 8), generator=generator)
    _, gemm_left = fp8_encode(gemm_left_source)
    _, gemm_right = fp8_encode(gemm_right_source)
    gemm_bias = torch.randn((5,), generator=generator)

    nvfp4_left_source = torch.randn((2, 32), generator=generator) * 1.5
    nvfp4_right_source = torch.randn((3, 32), generator=generator) * 1.25
    nvfp4_left = quantize_nvfp4(nvfp4_left_source)
    nvfp4_right = quantize_nvfp4(nvfp4_right_source)
    left_decoded = torch.tensor(nvfp4_left["dequantized"]).reshape(2, 32)
    right_decoded = torch.tensor(nvfp4_right["dequantized"]).reshape(3, 32)
    nvfp4_bias = torch.randn((3,), generator=generator)

    fp6_input = [0, 1, 15, 16, 31, 63, 64, 127, 128, 143, 159, 191, 207, 223, 239, 255]
    fp6_packed, fp6_unpacked = pack_fp6(fp6_input)

    rope_input = torch.randn((2, 8), generator=generator)
    rope_weights = torch.randn((8,), generator=generator) * 0.2 + 1
    rope_angles = torch.randn((2, 8), generator=generator) * 0.4
    rope_cosine = torch.cos(rope_angles)
    rope_sine = torch.sin(rope_angles)
    rope_norm = rope_input * torch.rsqrt(rope_input.square().mean(-1, keepdim=True)) * rope_weights
    rope_expected = torch.empty_like(rope_norm)
    rope_expected[:, 0::2] = -rope_norm[:, 1::2] * rope_sine[:, 0::2] + rope_norm[:, 0::2] * rope_cosine[:, 0::2]
    rope_expected[:, 1::2] = rope_norm[:, 0::2] * rope_sine[:, 1::2] + rope_norm[:, 1::2] * rope_cosine[:, 1::2]

    split_input = torch.randn((2, 8), generator=generator)
    split_weights = torch.randn((2, 4), generator=generator) * 0.2 + 1
    split_angles = torch.randn((2, 2, 2), generator=generator) * 0.4
    split_cosine = torch.cos(split_angles)
    split_sine = torch.sin(split_angles)
    split_norm = split_input * torch.rsqrt(split_input.square().mean(-1, keepdim=True) + 1e-6)
    split_expected = torch.empty_like(split_input)
    for row in range(2):
        for head in range(2):
            base = head * 4
            first = split_norm[row, base : base + 2] * split_weights[head, :2]
            second = split_norm[row, base + 2 : base + 4] * split_weights[head, 2:]
            split_expected[row, base : base + 2] = first * split_cosine[row, head] - second * split_sine[row, head]
            split_expected[row, base + 2 : base + 4] = second * split_cosine[row, head] + first * split_sine[row, head]

    rowwise_input = torch.randn((3, 9), generator=generator) * 2
    rowwise_scales = rowwise_input.abs().amax(-1) / 127.0
    rowwise_values = torch.round(rowwise_input / rowwise_scales[:, None]).clamp(-127, 127).to(torch.int8)

    block_input = torch.randn((2, 128), generator=generator) * 2
    block_plain = block_quantize(block_input, 128)
    block_gelu = block_quantize(block_input, 128, True)
    norm_scale = torch.randn((128,), generator=generator) * 0.05
    norm_shift = torch.randn((128,), generator=generator) * 0.05
    normalized = block_input * torch.rsqrt(block_input.square().mean(-1, keepdim=True) + 1e-5)
    normalized = normalized * (1 + norm_scale) + norm_shift
    block_norm = block_quantize(normalized, 128)

    fma_x = torch.randn((2, 128), generator=generator)
    fma_y = torch.randn((2, 128), generator=generator)
    fma_z = torch.randn((2, 128), generator=generator)
    fma_residual = fma_x + fma_y * fma_z
    fma_normalized = fma_residual * torch.rsqrt(fma_residual.square().mean(-1, keepdim=True))
    fma_quant = block_quantize(fma_normalized, 128)

    gated_input = torch.randn((3, 8), generator=generator)
    gate_logits = torch.randn((3, 2), generator=generator)
    gated_expected = gated_input.reshape(3, 2, 4) * (2 * torch.sigmoid(gate_logits))[:, :, None]

    lora_a = torch.randn((2, 32), generator=generator)
    lora_b = torch.randn((3, 2), generator=generator)
    lora_strength = 0.625
    lora_delta = ((lora_b.to(torch.bfloat16) * lora_strength) @ lora_a.to(torch.bfloat16)).to(torch.bfloat16)
    lora_merged = (right_decoded.to(torch.bfloat16) + lora_delta).to(torch.bfloat16).float()
    lora_fused = quantize_nvfp4(lora_merged)

    manifest = {
        "schema_version": 1,
        "fixture_revision": REVISION,
        "oracle": {
            "framework": "pytorch",
            "framework_version": EXPECTED_TORCH,
            "cuda_version": EXPECTED_CUDA,
            "seed": SEED,
        },
        "tolerances": {"fp32": FP32_TOLERANCE, "bf16_fp8": BF16_FP8_TOLERANCE},
        "native_kernel_inventory": KERNELS,
        "native_binding_inventory": BINDINGS,
        "cases": {
            "fused_add_round": {"delta": values(fused_delta), "weight": values(fused_weight), "seed": SEED, "expected": values(fused_delta + fused_weight)},
            "na3d": {"shape": list(na_shape), "kernel": [2, 2, 3], "query": values(query), "key": values(key), "value": values(value), "expected": values(na_expected)},
            "swiglu": {"rows": 3, "input_dim": 4, "output_dim": 5, "input": values(swiglu_input), "gate_weight": values(gate_weight), "up_weight": values(up_weight), "gate": values(gate), "up_mul_expected": values(gate * up_linear), "gate_up_expected": values(torch.nn.functional.silu(gate_linear) * up_linear)},
            "fp8_gemm": {"rows": 3, "output_dim": 5, "inner_dim": 8, "left": values(gemm_left), "right": values(gemm_right), "bias": values(gemm_bias), "expected": values(gemm_left @ gemm_right.T), "bias_expected": values(gemm_left @ gemm_right.T + gemm_bias)},
            "nvfp4": {"left_input": values(nvfp4_left_source), "left": nvfp4_left, "right_input": values(nvfp4_right_source), "right": nvfp4_right, "bias": values(nvfp4_bias), "scaled_mm_expected": values(left_decoded @ right_decoded.T + nvfp4_bias)},
            "scalars": {"left": -3.25, "right": 2.5, "product": -8.125, "amax_input": [-3.0, 0.0, 7.5, -2.0], "divisor": 2.5, "amax_expected": 3.0},
            "fp6": {"rows": 2, "columns": 8, "input": fp6_input, "packed": fp6_packed, "unpacked": fp6_unpacked},
            "rms_rope": {"rows": 2, "hidden_dim": 8, "input": values(rope_input), "weights": values(rope_weights), "cosine": values(rope_cosine), "sine": values(rope_sine), "expected": values(rope_expected)},
            "rms_split_rope": {"rows": 2, "heads": 2, "head_dim": 4, "input": values(split_input), "weights": values(split_weights), "cosine": values(split_cosine), "sine": values(split_sine), "expected": values(split_expected)},
            "rowwise_int8": {"rows": 3, "columns": 9, "input": values(rowwise_input), "values": rowwise_values.flatten().tolist(), "scales": values(rowwise_scales)},
            "blockwise": {"rows": 2, "columns": 128, "block_size": 128, "input": values(block_input), "plain": block_plain, "gelu": block_gelu, "norm_scale": values(norm_scale), "norm_shift": values(norm_shift), "norm": block_norm},
            "quant_rms_fma": {"rows": 2, "columns": 128, "block_size": 128, "x": values(fma_x), "y": values(fma_y), "z": values(fma_z), "residual": values(fma_residual), "quantized": fma_quant},
            "gated_attention": {"rows": 3, "heads": 2, "head_dim": 4, "input": values(gated_input), "gate_logits": values(gate_logits), "expected": values(gated_expected)},
            "lora": {"rank": 2, "strength": lora_strength, "factor_a": values(lora_a), "factor_b": values(lora_b), "fused_dequantized": lora_fused["dequantized"], "unfused_dequantized": nvfp4_right["dequantized"]},
        },
    }
    serialized = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    (args.output / "manifest.json").write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    print(json.dumps({"fixture_revision": REVISION, "manifest_sha256": digest, "result": "pass"}, sort_keys=True))


if __name__ == "__main__":
    main()
