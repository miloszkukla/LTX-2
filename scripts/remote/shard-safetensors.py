#!/usr/bin/env python3
"""Split one safetensors file without memory-mapping its full payload.

This is intended for disk-streamed checkpoints whose single-file virtual mapping is
larger than the worker's cgroup memory limit. Tensor bytes are copied unchanged and
each output shard remains a normal independently mmap-able safetensors file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import BinaryIO

HEADER_PREFIX_BYTES = 8
COPY_BUFFER_BYTES = 16 * 1024 * 1024
ASSET_KEYS = {"tokenizer_json"}
ASSET_PREFIX = "hf_asset__"


def copy_exact(source: BinaryIO, target: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        chunk = source.read(min(remaining, COPY_BUFFER_BYTES))
        if not chunk:
            raise EOFError(f"source checkpoint ended with {remaining} bytes left to copy")
        target.write(chunk)
        remaining -= len(chunk)


def encoded_header(tensors: list[tuple[str, dict]], metadata: dict | None) -> bytes:
    offset = 0
    body: dict[str, object] = {}
    if metadata is not None:
        body["__metadata__"] = metadata
    for key, info in tensors:
        size = info["data_offsets"][1] - info["data_offsets"][0]
        body[key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    padding = (-len(raw)) % 8
    return raw + (b" " * padding)


def group_tensors(tensors: list[tuple[str, dict]], target_bytes: int) -> list[list[tuple[str, dict]]]:
    assets = [item for item in tensors if item[0] in ASSET_KEYS or item[0].startswith(ASSET_PREFIX)]
    weights = [item for item in tensors if item not in assets]
    groups: list[list[tuple[str, dict]]] = [[]]
    group_bytes = 0

    for item in (*assets, *weights):
        size = item[1]["data_offsets"][1] - item[1]["data_offsets"][0]
        if groups[-1] and group_bytes + size > target_bytes:
            groups.append([])
            group_bytes = 0
        groups[-1].append(item)
        group_bytes += size
    return groups


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-gib", type=float, default=8.0)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file() or source.suffix != ".safetensors":
        raise SystemExit("source must be an existing .safetensors file")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    if args.target_gib <= 0:
        raise SystemExit("--target-gib must be positive")

    with source.open("rb") as handle:
        header_length = int.from_bytes(handle.read(HEADER_PREFIX_BYTES), "little")
        header = json.loads(handle.read(header_length))
    metadata = header.pop("__metadata__", None)
    tensors = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
    groups = group_tensors(tensors, int(args.target_gib * 1024**3))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_start = HEADER_PREFIX_BYTES + header_length
    shard_records = []

    with source.open("rb") as source_handle:
        for index, group in enumerate(groups, start=1):
            name = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            destination = output_dir / name
            partial = destination.with_suffix(destination.suffix + ".partial")
            shard_header = encoded_header(group, metadata)
            with partial.open("xb") as target_handle:
                target_handle.write(len(shard_header).to_bytes(HEADER_PREFIX_BYTES, "little"))
                target_handle.write(shard_header)
                for _key, info in group:
                    start, end = info["data_offsets"]
                    source_handle.seek(data_start + start)
                    copy_exact(source_handle, target_handle, end - start)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            partial.replace(destination)
            shard_records.append(
                {
                    "path": name,
                    "size_bytes": destination.stat().st_size,
                    "tensor_count": len(group),
                    "sha256": sha256(destination),
                }
            )
            sys.stdout.write(f"wrote {destination} ({destination.stat().st_size} bytes)\n")
            sys.stdout.flush()

    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "target_gib": args.target_gib,
        "tensor_count": len(tensors),
        "shard_count": len(shard_records),
        "shards": shard_records,
        "tensor_payloads_byte_identical": True,
    }
    (output_dir / "shards.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
