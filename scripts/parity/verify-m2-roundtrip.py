#!/usr/bin/env python3
"""Verify C#-written M2 safetensors files with the pinned Python implementation."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def assert_same(source: Path, candidate: Path) -> None:
    expected = load_file(source, device="cpu")
    actual = load_file(candidate, device="cpu")
    if expected.keys() != actual.keys():
        raise SystemExit(f"tensor key mismatch: {source.name} vs {candidate.name}")
    for key in expected:
        if expected[key].dtype != actual[key].dtype or expected[key].shape != actual[key].shape:
            raise SystemExit(f"tensor layout mismatch: {candidate.name}:{key}")
        if not torch.equal(expected[key], actual[key]):
            raise SystemExit(f"tensor payload mismatch: {candidate.name}:{key}")
    if metadata(source) != metadata(candidate):
        raise SystemExit(f"metadata mismatch: {source.name} vs {candidate.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    pairs = [
        ("storage-mixed.safetensors", "storage-roundtrip.safetensors"),
        ("lora.safetensors", "lora-roundtrip.safetensors"),
        ("storage-mixed.safetensors", "torch-roundtrip.safetensors"),
    ]
    for source_name, candidate_name in pairs:
        assert_same(args.fixture_dir / source_name, args.output_dir / candidate_name)

    result = {
        "result": "pass",
        "checks": len(pairs),
        "message": "Python safetensors accepted all C# round trips with exact layouts, payloads, and metadata.",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
