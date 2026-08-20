#!/usr/bin/env bash
set -Eeuo pipefail

report_path=""
if [[ ${1:-} == "--report" ]]; then
    report_path=${2:?missing report path}
    shift 2
fi
if (($# != 0)); then
    echo "usage: $0 [--report PATH]" >&2
    exit 2
fi
if ((EUID != 0)); then
    echo "bootstrap-host.sh must run as root" >&2
    exit 1
fi

source /etc/os-release
if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 24.04 ]]; then
    echo "Ubuntu 24.04 is required" >&2
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo "linux/amd64 is required" >&2
    exit 1
fi

bootstrap_temp=$(mktemp -d /tmp/ltx-bootstrap.XXXXXX)
cleanup() {
    if [[ $bootstrap_temp == /tmp/ltx-bootstrap.* && -d $bootstrap_temp ]]; then
        rm -rf -- "$bootstrap_temp"
    fi
}
trap cleanup EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git gnupg jq \
    libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev \
    libgl1 libglib2.0-0 libopenimageio-dev libopencv-dev \
    libswresample-dev libswscale-dev ninja-build openimageio-tools pkg-config skopeo \
    python3 python3-dev python3-pip python3-venv software-properties-common \
    unzip

if ! command -v dotnet >/dev/null || [[ $(dotnet --version) != 10.* ]]; then
    curl -fsSLo "$bootstrap_temp/packages-microsoft-prod.deb" \
        https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb
    dpkg -i "$bootstrap_temp/packages-microsoft-prod.deb"
    apt-get update
    apt-get install -y --no-install-recommends dotnet-sdk-10.0 gh
elif ! command -v gh >/dev/null; then
    apt-get install -y --no-install-recommends gh
fi

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

ldconfig
dotnet --info >/dev/null
nvcc --version >/dev/null
ffmpeg -version >/dev/null
oiiotool --help >/dev/null

if [[ -n $report_path ]]; then
    mkdir -p "$(dirname "$report_path")"
    REPORT_PATH="$report_path" python3 - <<'PY'
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

packages = [
    "cmake", "dotnet-sdk-10.0", "ffmpeg", "gh", "git", "libavcodec-dev",
    "libavformat-dev", "libavutil-dev", "libopenimageio-dev", "libopencv-dev",
    "ninja-build", "python3",
]

def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()

versions = {}
for package in packages:
    try:
        versions[package] = command("dpkg-query", "-W", "-f=${Version}", package)
    except (subprocess.CalledProcessError, FileNotFoundError):
        versions[package] = "unavailable"

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "platform": {"system": platform.system(), "machine": platform.machine()},
    "packages": versions,
    "tools": {
        "dotnet": command("dotnet", "--version"),
        "uv": command("uv", "--version"),
    },
}
Path(os.environ["REPORT_PATH"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
fi

echo "remote bootstrap complete"
