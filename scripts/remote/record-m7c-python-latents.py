#!/usr/bin/env python3
"""Record the two reference diffusion-stage outputs for M7C port diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.utils import to_velocity
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.model_paths import ModelPaths


PROMPT = (
    "Two adult women with long blonde hair play a lively beach-volleyball rally on a sunny beach. "
    "They wear sporty bikinis appropriate for beach volleyball and speak casually to each other between plays, "
    "smiling and calling the ball. Natural, realistic athletic motion; ocean waves, warm sand, gentle sea breeze, "
    "distant beach ambience, synchronized dialogue and sound. Cinematic tracking camera, no cuts."
)
CHECKPOINT = "/workspace/ltx-model-cache/M7A/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors"
UPSCALER = "/workspace/ltx-model-cache/M7C/ltx23-spatial-upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = "/workspace/ltx-model-cache/M7C/ltx23-gemma3-assets"
OUTPUT = Path("/workspace/LTX-2/build/M7C/diagnostics/python-stage-latents.safetensors")


class DetailedRecordingDiffusionStage:
    """Record final states and the first production-geometry transformer call per stage."""

    def __init__(self, stage: object) -> None:
        self._stage = stage
        self.video_states = []
        self.audio_states = []
        self.step_tensors: dict[str, torch.Tensor] = {}

    def __call__(self, *args: object, **kwargs: object):
        prefix = f"stage{len(self.video_states) + 1}"
        denoiser = kwargs["denoiser"]
        recorded = False

        def recording_denoiser(transformer, video_state, audio_state, sigmas, step_index):
            nonlocal recorded
            hooks = []
            if not recorded and prefix == "stage1":
                velocity_model = transformer._model.velocity_model
                block = velocity_model.transformer_blocks[0]

                def block_input_hook(_module, _args, hook_kwargs):
                    video_args = hook_kwargs["video"]
                    audio_args = hook_kwargs["audio"]
                    self.step_tensors["stage1_step0_block0_video_input"] = (
                        video_args.x.detach().cpu().contiguous()
                    )
                    self.step_tensors["stage1_step0_block0_audio_input"] = (
                        audio_args.x.detach().cpu().contiguous()
                    )

                def block_output_hook(_module, _args, _kwargs, output):
                    self.step_tensors["stage1_step0_block0_video_output"] = (
                        output[0].x.detach().cpu().contiguous()
                    )
                    self.step_tensors["stage1_step0_block0_audio_output"] = (
                        output[1].x.detach().cpu().contiguous()
                    )

                def tensor_output_hook(name):
                    def capture(_module, _args, _kwargs, output):
                        self.step_tensors[f"stage1_step0_block0_{name}"] = output.detach().cpu().contiguous()

                    return capture

                hooks.append(block.register_forward_pre_hook(block_input_hook, with_kwargs=True))
                hooks.append(block.register_forward_hook(block_output_hook, with_kwargs=True))
                for name in (
                    "attn1",
                    "attn2",
                    "audio_attn1",
                    "audio_attn2",
                    "audio_to_video_attn",
                    "video_to_audio_attn",
                    "ff",
                    "audio_ff",
                ):
                    hooks.append(
                        getattr(block, name).register_forward_hook(tensor_output_hook(name), with_kwargs=True)
                    )
                attn1 = block.attn1
                for name, module in (
                    ("attn1_to_q", attn1.to_q),
                    ("attn1_to_k", attn1.to_k),
                    ("attn1_q_norm", attn1.q_norm),
                    ("attn1_k_norm", attn1.k_norm),
                    ("attn1_to_v", attn1.to_v),
                ):
                    hooks.append(module.register_forward_hook(tensor_output_hook(name), with_kwargs=True))

                def attended_input_hook(_module, hook_args, _hook_kwargs):
                    self.step_tensors["stage1_step0_block0_attn1_attended"] = (
                        hook_args[0].detach().cpu().contiguous()
                    )

                hooks.append(attn1.to_out[0].register_forward_pre_hook(attended_input_hook, with_kwargs=True))
                original_preattention = attn1.preattention_function

                def recording_preattention(q, k, attn_module, mask, pe, k_pe):
                    resolved_q, resolved_k = original_preattention(q, k, attn_module, mask, pe, k_pe)
                    self.step_tensors["stage1_step0_block0_attn1_q_rope"] = (
                        resolved_q.detach().cpu().contiguous()
                    )
                    self.step_tensors["stage1_step0_block0_attn1_k_rope"] = (
                        resolved_k.detach().cpu().contiguous()
                    )
                    return resolved_q, resolved_k

                attn1.preattention_function = recording_preattention
            video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_index)
            for hook in hooks:
                hook.remove()
            if not recorded and prefix == "stage1":
                attn1.preattention_function = original_preattention
            if not recorded:
                if video_state is not None and video_result is not None:
                    self.step_tensors[f"{prefix}_step0_video_input"] = video_state.latent.detach().cpu().contiguous()
                    self.step_tensors[f"{prefix}_step0_video_velocity"] = to_velocity(
                        video_state.latent, sigmas[step_index], video_result.denoised
                    ).detach().cpu().contiguous()
                if audio_state is not None and audio_result is not None:
                    self.step_tensors[f"{prefix}_step0_audio_input"] = audio_state.latent.detach().cpu().contiguous()
                    self.step_tensors[f"{prefix}_step0_audio_velocity"] = to_velocity(
                        audio_state.latent, sigmas[step_index], audio_result.denoised
                    ).detach().cpu().contiguous()
                recorded = True
            return video_result, audio_result

        kwargs["denoiser"] = recording_denoiser
        video_state, audio_state = self._stage(*args, **kwargs)
        if video_state is not None:
            self.video_states.append(video_state)
            self.audio_states.append(audio_state)
        return video_state, audio_state

    def __getattr__(self, name: str) -> object:
        return getattr(self._stage, name)


def stats(value: torch.Tensor) -> dict[str, object]:
    resolved = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": resolved.min().item(),
        "maximum": resolved.max().item(),
        "mean": resolved.mean().item(),
        "standard_deviation": resolved.std().item(),
        "finite": bool(torch.isfinite(resolved).all().item()),
    }


@torch.inference_mode()
def main() -> None:
    pipeline = DistilledPipeline(
        model_paths=ModelPaths.from_monolith(CHECKPOINT, GEMMA),
        spatial_upsampler_path=UPSCALER,
        loras=[],
    )
    recorder = DetailedRecordingDiffusionStage(pipeline.stage)
    pipeline.stage = recorder
    video, _audio, _frames, _tiling = pipeline(
        prompt=PROMPT,
        seed=20260821,
        height=1024,
        width=1536,
        num_frames=241,
        frame_rate=24,
        images=[],
    )
    if len(recorder.video_states) != 2 or len(recorder.audio_states) != 2:
        raise RuntimeError("The reference pipeline did not record exactly two diffusion stages.")
    tensors = {
        "stage1_video": recorder.video_states[0].latent.detach().cpu().contiguous(),
        "stage1_audio": recorder.audio_states[0].latent.detach().cpu().contiguous(),
        "stage2_video": recorder.video_states[1].latent.detach().cpu().contiguous(),
        "stage2_audio": recorder.audio_states[1].latent.detach().cpu().contiguous(),
        **recorder.step_tensors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        OUTPUT,
        metadata={
            "fixture_revision": "m7c-python-two-stage-latents-v1",
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "seed": "20260821",
            "frames": "241",
            "frame_rate": "24",
            "width": "1536",
            "height": "1024",
        },
    )
    del video
    print(json.dumps({name: stats(value) for name, value in tensors.items()}, indent=2))


if __name__ == "__main__":
    main()
