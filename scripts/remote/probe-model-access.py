#!/usr/bin/env python3
"""Probe required Hugging Face artifacts without downloading model content."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def token_from_environment() -> tuple[str | None, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "interactive_environment"
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"HF_TOKEN="):
                value = item.split(b"=", 1)[1].decode()
                if value:
                    return value, "pid1_transient"
    except (OSError, UnicodeDecodeError):
        pass
    return None, "absent"


def fetch_model(repo: str, revision: str, token: str) -> dict[str, object]:
    encoded_repo = urllib.parse.quote(repo, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_revision}?blobs=true"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "ltx-csharp-m0/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Hugging Face probe failed for {repo}@{revision}: HTTP {error.code}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/M0/artifact-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/model-access.json")
    args = parser.parse_args()

    token, token_source = token_from_environment()
    if not token:
        raise SystemExit("HF_TOKEN is absent; authenticated read-only model probe is required")

    manifest = json.loads(args.manifest.read_text())
    cache: dict[tuple[str, str], dict[str, object]] = {}
    results: list[dict[str, object]] = []
    phase_sizes: dict[str, int] = defaultdict(int)
    common_size = 0

    for artifact in manifest["artifacts"]:
        key = (artifact["repo"], artifact["revision"])
        if key not in cache:
            cache[key] = fetch_model(*key, token)
        info = cache[key]
        selected = [
            sibling
            for sibling in info.get("siblings", [])
            if fnmatch.fnmatch(str(sibling.get("rfilename", "")), artifact["selector"])
        ]
        if not selected:
            raise SystemExit(f"required selector missing: {artifact['repo']}:{artifact['selector']}")

        files = []
        artifact_size = 0
        for sibling in selected:
            lfs = sibling.get("lfs") or {}
            size = int(lfs.get("size") or sibling.get("size") or 0)
            sha256 = lfs.get("sha256") or lfs.get("oid")
            if sha256 and str(sha256).startswith("sha256:"):
                sha256 = str(sha256).split(":", 1)[1]
            checksum_source = "huggingface_lfs_metadata" if sha256 else "repository_revision"
            files.append(
                {
                    "path": sibling["rfilename"],
                    "size_bytes": size,
                    "sha256": sha256,
                    "checksum_source": checksum_source,
                }
            )
            artifact_size += size
        result = {
            "id": artifact["id"],
            "repo": artifact["repo"],
            "requested_revision": artifact["revision"],
            "resolved_revision": info.get("sha"),
            "selector": artifact["selector"],
            "phase": artifact["phase"],
            "required": artifact["required"],
            "access": "authenticated_read_succeeded",
            "size_bytes": artifact_size,
            "files": files,
        }
        results.append(result)
        if artifact["phase"] == "ltx25-common":
            common_size += artifact_size
        else:
            phase_sizes[artifact["phase"]] += artifact_size

    if common_size:
        for phase in ("ltx25-dev-standard", "ltx25-distilled-standard"):
            phase_sizes[phase] += common_size
    reserve = int(manifest["cache_and_fixture_reserve_bytes"])
    peak_models = max(phase_sizes.values(), default=0)
    required_disk = peak_models + reserve
    required_with_headroom = (required_disk * 120 + 99) // 100
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hf_token": {"present": True, "scope_expected": "fine-grained/read-only", "source": token_source},
        "probe": "authenticated metadata-only; no model bytes downloaded",
        "artifacts": results,
        "disk_gate": {
            "cache_eviction_between_phases": True,
            "phase_model_bytes": dict(sorted(phase_sizes.items())),
            "peak_model_bytes": peak_models,
            "cache_and_fixture_reserve_bytes": reserve,
            "required_disk_bytes": required_disk,
            "required_disk_with_20_percent_headroom_bytes": required_with_headroom,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"model access probes passed: {len(results)} artifacts; required disk={required_with_headroom} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
