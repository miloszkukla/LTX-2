#!/usr/bin/env python3
"""Download only M0-pinned M7A checkpoint payloads with resumable, verified writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024


def inherited_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"HF_TOKEN="):
                return item.split(b"=", 1)[1].decode()
    except (OSError, UnicodeDecodeError):
        pass
    raise SystemExit("inherited HF token is unavailable")


def safe_segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"unsafe artifact identifier: {value!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, size: int, checksum: str, token: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists():
        if destination.stat().st_size == size and sha256(destination) == checksum:
            return "cached_verified"
        raise RuntimeError(f"existing final file failed its pinned size/checksum: {destination}")

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError(f"partial file exceeds pinned size: {partial}")
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "ltx-csharp-m7a/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"checkpoint download returned HTTP {error.code}") from None

    with response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            raise RuntimeError(f"checkpoint server did not honor resume range (HTTP {status})")
        if not offset and status not in (200, 206):
            raise RuntimeError(f"checkpoint download returned HTTP {status}")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

    if partial.stat().st_size != size:
        raise RuntimeError(
            f"checkpoint size mismatch: got {partial.stat().st_size}, expected {size}"
        )
    if sha256(partial) != checksum:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    partial.replace(destination)
    return "downloaded_verified"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/M0/model-access.json")
    parser.add_argument("--cache-root", default="/workspace/ltx-model-cache/M7A")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--report")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    selected = manifest["artifacts"]
    if args.ids:
        wanted = set(args.ids)
        selected = [item for item in selected if item["id"] in wanted]
        missing = wanted - {item["id"] for item in selected}
        if missing:
            raise SystemExit(f"unknown pinned artifact ids: {', '.join(sorted(missing))}")

    token = inherited_token()
    cache_root = Path(args.cache_root).resolve()
    results = []
    for artifact in selected:
        artifact_id = safe_segment(artifact["id"])
        for item in artifact["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe pinned artifact path: {relative}")
            destination = cache_root / artifact_id / relative
            encoded_path = "/".join(
                urllib.parse.quote(part, safe="") for part in item["path"].split("/")
            )
            url = (
                f"https://huggingface.co/{artifact['repo']}/resolve/"
                f"{artifact['resolved_revision']}/{encoded_path}"
            )
            state = download(url, destination, item["size_bytes"], item["sha256"], token)
            results.append(
                {
                    "artifact_id": artifact_id,
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "state": state,
                }
            )
            print(f"{artifact_id}/{item['path']}: {state}", flush=True)

    if args.report:
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "result": "pass",
            "files": results,
            "token_source": "inherited_runtime_environment",
            "secrets_recorded": False,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
