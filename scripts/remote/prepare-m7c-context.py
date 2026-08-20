#!/usr/bin/env python3
"""Prepare the accepted C# pipeline's prompt-conditioning input for M7C."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_pipelines.utils.blocks import PromptEncoder
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--gemma-revision", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("M7C prompt conditioning requires exactly one CUDA device")

    checkpoint = str(Path(args.checkpoint).resolve())
    gemma_root = str(Path(args.gemma_root).resolve())
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = PromptEncoder(
        ModelPaths.from_monolith(checkpoint, gemma_root),
        torch.bfloat16,
        torch.device("cuda"),
        offload_mode=OffloadMode.NONE,
    )
    (context,) = encoder([args.prompt])
    tensors = {
        "video_context": context.video_encoding.detach().cpu().contiguous(),
        "audio_context": context.audio_encoding.detach().cpu().contiguous(),
    }
    elapsed = time.perf_counter() - started
    prompt_sha256 = hashlib.sha256(args.prompt.encode()).hexdigest()
    save_file(
        tensors,
        str(output),
        metadata={
            "schema_version": "1",
            "prompt_sha256": prompt_sha256,
            "checkpoint_revision": args.checkpoint_revision,
            "gemma_revision": args.gemma_revision,
            "execution": "shared_prompt_conditioning_for_native_csharp",
        },
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result": "pass",
        "execution": "shared_prompt_conditioning_for_native_csharp",
        "offload": "none",
        "prompt": args.prompt,
        "prompt_sha256": prompt_sha256,
        "checkpoint_revision": args.checkpoint_revision,
        "gemma_revision": args.gemma_revision,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        },
        "output_path": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "secrets_recorded": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": "pass", "output": str(output), "elapsed_seconds": elapsed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
