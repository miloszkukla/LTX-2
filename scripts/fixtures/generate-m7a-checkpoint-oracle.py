#!/usr/bin/env python3
"""Generate the retained Python numerical oracle for the M7A production checkpoint runtime."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_core.loader import SingleGPUModelBuilder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.transformer import LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_core.model.transformer.attention import PytorchAttention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer import TransformerOpsConfig
from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator


class MathLTXModelConfigurator(LTXModelConfigurator):
    """Build the official model with an explicitly retained portable attention backend."""

    @classmethod
    def from_metadata(cls, metadata):
        attention = PytorchAttention(priority=[SDPBackend.MATH])
        ops = TransformerOpsConfig.from_functions(attention=attention, masked_attention=attention)
        return super().from_metadata(metadata, ops=ops)


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260820)
    torch.cuda.manual_seed_all(20260820)

    with safe_open(args.contexts, framework="pt", device=str(device)) as source:
        video_context = source.get_tensor("video_context")
        audio_context = source.get_tensor("audio_context")

    # One token per modality still traverses every real checkpoint block, including text and
    # audio/video cross attention, while making this correctness fixture independent of output size.
    video_latent = torch.randn((1, 1, 128), device=device, dtype=dtype)
    audio_latent = torch.randn((1, 1, 128), device=device, dtype=dtype)
    sigma = torch.tensor([0.725], device=device, dtype=torch.float32)
    timesteps = sigma[:, None]
    video_positions = torch.tensor([[[[0.0, 0.125]], [[0.0, 32.0]], [[0.0, 32.0]]]], device=device)
    audio_positions = torch.zeros((1, 1, 1, 2), device=device, dtype=torch.float32)
    video = Modality(video_latent, sigma, timesteps, video_positions, video_context)
    audio = Modality(audio_latent, sigma, timesteps, audio_positions, audio_context)

    transformer = SingleGPUModelBuilder(
        model_class_configurator=MathLTXModelConfigurator,
        model_path=args.checkpoint,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=device, dtype=dtype).eval()
    transformer_trace: dict[str, torch.Tensor] = {}

    def capture_input(_module, _args, kwargs):
        for modality_name in ("video", "audio"):
            prepared = kwargs[modality_name]
            transformer_trace[f"transformer_{modality_name}_prepared"] = cpu(prepared.x)
            transformer_trace[f"transformer_{modality_name}_timestep"] = cpu(prepared.timesteps)
            transformer_trace[f"transformer_{modality_name}_embedded_timestep"] = cpu(prepared.embedded_timestep)
            transformer_trace[f"transformer_{modality_name}_prompt_timestep"] = cpu(prepared.prompt_timestep)

    def capture_output(layer: int):
        def hook(_module, _args, output):
            transformer_trace[f"transformer_video_block_{layer}"] = cpu(output[0].x)
            transformer_trace[f"transformer_audio_block_{layer}"] = cpu(output[1].x)

        return hook

    handles = [transformer.transformer_blocks[0].register_forward_pre_hook(capture_input, with_kwargs=True)]
    handles.extend(
        block.register_forward_hook(capture_output(layer))
        for layer, block in enumerate(transformer.transformer_blocks)
    )
    diagnostic_modules = {
        "attn1": transformer.transformer_blocks[0].attn1,
        "attn2": transformer.transformer_blocks[0].attn2,
        "audio_attn1": transformer.transformer_blocks[0].audio_attn1,
        "audio_attn2": transformer.transformer_blocks[0].audio_attn2,
        "audio_to_video_attn": transformer.transformer_blocks[0].audio_to_video_attn,
        "video_to_audio_attn": transformer.transformer_blocks[0].video_to_audio_attn,
        "ff": transformer.transformer_blocks[0].ff,
        "audio_ff": transformer.transformer_blocks[0].audio_ff,
    }

    def capture_module(name: str):
        def hook(_module, _args, output):
            transformer_trace[f"transformer_block_0_{name}"] = cpu(output)

        return hook

    handles.extend(module.register_forward_hook(capture_module(name)) for name, module in diagnostic_modules.items())

    def capture_attention_input(_module, args, _kwargs):
        transformer_trace["transformer_block_0_attn1.input"] = cpu(args[0])

    handles.append(
        transformer.transformer_blocks[0].attn1.register_forward_pre_hook(capture_attention_input, with_kwargs=True)
    )
    def capture_text_attention_input(_module, args, kwargs):
        transformer_trace["transformer_block_0_attn2.input"] = cpu(args[0])
        transformer_trace["transformer_block_0_attn2.context"] = cpu(kwargs["context"])

    handles.append(
        transformer.transformer_blocks[0].attn2.register_forward_pre_hook(
            capture_text_attention_input, with_kwargs=True
        )
    )
    def capture_text_attention_output_input(_module, args):
        transformer_trace["transformer_block_0_attn2.attended"] = cpu(args[0])

    handles.append(
        transformer.transformer_blocks[0].attn2.to_out[0].register_forward_pre_hook(
            capture_text_attention_output_input
        )
    )
    def capture_named(name: str):
        def hook(_module, _args, output):
            transformer_trace[f"transformer_{name}"] = cpu(output)

        return hook

    for name, module in {
        "attn1.to_v": transformer.transformer_blocks[0].attn1.to_v,
        "attn1.to_gate_logits": transformer.transformer_blocks[0].attn1.to_gate_logits,
        "attn1.to_out": transformer.transformer_blocks[0].attn1.to_out[0],
        "attn2.to_q": transformer.transformer_blocks[0].attn2.to_q,
        "attn2.to_k": transformer.transformer_blocks[0].attn2.to_k,
        "attn2.to_v": transformer.transformer_blocks[0].attn2.to_v,
        "attn2.q_norm": transformer.transformer_blocks[0].attn2.q_norm,
        "attn2.k_norm": transformer.transformer_blocks[0].attn2.k_norm,
        "attn2.to_gate_logits": transformer.transformer_blocks[0].attn2.to_gate_logits,
        "attn2.to_out": transformer.transformer_blocks[0].attn2.to_out[0],
    }.items():
        handles.append(module.register_forward_hook(capture_module(name)))
    for name, module in {
        "video_adaln_time_projection": transformer.adaln_single.emb.time_proj,
        "video_adaln_linear_1": transformer.adaln_single.emb.timestep_embedder.linear_1,
        "video_adaln_embed_silu": transformer.adaln_single.emb.timestep_embedder.act,
        "video_adaln_embedded": transformer.adaln_single.emb.timestep_embedder.linear_2,
        "video_adaln_silu": transformer.adaln_single.silu,
        "video_adaln_modulation": transformer.adaln_single.linear,
    }.items():
        handles.append(module.register_forward_hook(capture_named(name)))
    handles.append(transformer.norm_out.register_forward_hook(capture_named("video_output_norm")))

    def capture_video_projection_input(_module, args):
        transformer_trace["transformer_video_output_projection_input"] = cpu(args[0])

    handles.append(transformer.proj_out.register_forward_pre_hook(capture_video_projection_input))
    # Pin the portable math backend so the retained cross-language oracle does not depend on
    # whichever architecture-specific flash kernel happens to win PyTorch's dispatcher.
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH], set_priority=True):
        video_velocity, audio_velocity = transformer(video, audio, None)
    for handle in handles:
        handle.remove()

    tensors = {
        "video_latent": cpu(video_latent),
        "audio_latent": cpu(audio_latent),
        "sigma": cpu(sigma),
        "timesteps": cpu(timesteps),
        "video_positions": cpu(video_positions),
        "audio_positions": cpu(audio_positions),
        "video_context": cpu(video_context),
        "audio_context": cpu(audio_context),
        "video_velocity": cpu(video_velocity),
        "audio_velocity": cpu(audio_velocity),
        **transformer_trace,
    }
    del transformer, video_velocity, audio_velocity, video, audio
    gc.collect()
    torch.cuda.empty_cache()

    video_latent_5d = torch.randn((1, 128, 1, 1, 1), device=device, dtype=dtype)
    video_decoder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=args.checkpoint,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        decoded_video = video_decoder(video_latent_5d)
    tensors["video_decoder_latent"] = cpu(video_latent_5d)
    tensors["decoded_video"] = cpu(decoded_video)
    del video_decoder, decoded_video, video_latent_5d
    gc.collect()
    torch.cuda.empty_cache()

    audio_latent_4d = torch.randn((1, 8, 2, 16), device=device, dtype=dtype)
    audio_decoder = SingleGPUModelBuilder(
        model_class_configurator=AudioDecoderConfigurator,
        model_path=args.checkpoint,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        decoded_audio = audio_decoder(audio_latent_4d)
    tensors["audio_decoder_latent"] = cpu(audio_latent_4d)
    tensors["decoded_audio"] = cpu(decoded_audio)
    del audio_decoder, audio_latent_4d
    gc.collect()
    torch.cuda.empty_cache()

    vocoder = SingleGPUModelBuilder(
        model_class_configurator=VocoderConfigurator,
        model_path=args.checkpoint,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        decoded_waveform = vocoder(decoded_audio)
    tensors["decoded_waveform"] = cpu(decoded_waveform)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1",
        "fixture_revision": "m7a-real-checkpoint-runtime-v4",
        "framework": torch.__version__,
        "checkpoint_model_version": "2.3.0",
        "seed": "20260820",
        "scope": "full_48_layer_av_transformer_conv_video_vae_audio_vae_vocoder_bwe",
        "attention_backend": "pytorch_sdpa_math",
    }
    save_file(tensors, output, metadata=metadata)
    manifest = {
        "schema_version": 1,
        "fixture_revision": metadata["fixture_revision"],
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "tolerances": {"bf16": {"rtol": 0.02, "atol": 0.005}},
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
