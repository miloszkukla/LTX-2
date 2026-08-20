#!/usr/bin/env python3
"""Verify the NVIDIA base digest and select public GHCR or the checked-in fallback."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NVIDIA_REPO = "nvidia/cuda"
NVIDIA_TAG = "13.2.0-cudnn-devel-ubuntu24.04"
EXPECTED_AMD64 = "sha256:293837d7ad950f195e6ba5e3c0478c50b8c80349a70ad14687f39ed8ff0a8e4a"
GHCR_IMAGE = "ghcr.io/miloszkukla/ltx-csharp-base"


def nvidia_digest() -> tuple[str, str]:
    params = urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": f"repository:{NVIDIA_REPO}:pull"}
    )
    with urllib.request.urlopen(f"https://auth.docker.io/token?{params}", timeout=30) as response:
        token = json.load(response)["token"]
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{NVIDIA_REPO}/manifests/{NVIDIA_TAG}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        index = json.load(response)
        index_digest = response.headers["Docker-Content-Digest"]
    platform_digest = next(
        item["digest"]
        for item in index["manifests"]
        if item.get("platform", {}).get("os") == "linux"
        and item.get("platform", {}).get("architecture") == "amd64"
    )
    return index_digest, platform_digest


def ghcr_status() -> tuple[str, str | None, str]:
    url = f"https://ghcr.io/v2/miloszkukla/ltx-csharp-base/manifests/cuda13.2-dotnet10"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read(1)
            return "available", response.headers.get("Docker-Content-Digest"), f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return "unavailable", None, f"HTTP {error.code} {error.reason} on anonymous manifest request"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/ghcr.json")
    args = parser.parse_args()
    index_digest, amd64_digest = nvidia_digest()
    if amd64_digest != EXPECTED_AMD64:
        raise SystemExit(f"NVIDIA linux/amd64 digest changed: {amd64_digest}")
    status, digest, detail = ghcr_status()
    if status == "available":
        selection = "ghcr_digest_pending_fresh_worker_pull"
        accepted = False
        failure = "Anonymous manifest succeeded, but a fresh-worker pull by digest has not been recorded."
    else:
        selection = "nvidia_base_plus_checked_in_bootstrap"
        accepted = True
        failure = detail
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "nvidia_base": {
            "tag": f"{NVIDIA_REPO}:{NVIDIA_TAG}",
            "index_digest": index_digest,
            "linux_amd64_digest": amd64_digest,
            "verified": True,
        },
        "ghcr": {
            "image": GHCR_IMAGE,
            "digest": digest,
            "anonymous_manifest": status,
            "workflow": ".github/workflows/remote-base.yml",
            "personal_token_used": False,
        },
        "accepted": accepted,
        "worker_strategy": selection,
        "failure": failure,
        "fallback": {
            "base": f"{NVIDIA_REPO}@{amd64_digest}",
            "bootstrap": "scripts/remote/bootstrap-host.sh",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not accepted:
        raise SystemExit(failure)
    print(f"GHCR fallback selected: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
