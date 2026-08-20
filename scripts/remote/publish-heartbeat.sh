#!/usr/bin/env bash
set -Eeuo pipefail

milestone=${1:?usage: publish-heartbeat.sh MILESTONE PHASE [FAILURE_CODE]}
phase=${2:?usage: publish-heartbeat.sh MILESTONE PHASE [FAILURE_CODE]}
failure=${3:-none}
if [[ ! $milestone =~ ^M[0-7]$ || ! $phase =~ ^[A-Za-z0-9_.:-]{1,80}$ || ! $failure =~ ^[A-Za-z0-9_.:-]{1,160}$ ]]; then
    echo "heartbeat fields must be short redacted identifiers" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
heartbeat_temp=$(mktemp -d /tmp/ltx-heartbeat.XXXXXX)
cleanup() {
    if [[ $heartbeat_temp == /tmp/ltx-heartbeat.* && -d $heartbeat_temp ]]; then
        rm -rf -- "$heartbeat_temp"
    fi
}
trap cleanup EXIT

source_commit=$(git -C "$repo_root" rev-parse HEAD)
heartbeat_file="$heartbeat_temp/heartbeat.json"
MILESTONE="$milestone" \
PHASE="$phase" \
FAILURE="$failure" \
SOURCE_COMMIT="$source_commit" \
HEARTBEAT_FILE="$heartbeat_file" \
python3 - <<'PY'
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

instance_id = "redacted"
try:
    values = {}
    for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            name, value = item.split(b"=", 1)
            if name in {b"CONTAINER_ID", b"VAST_CONTAINERLABEL"}:
                values[name.decode()] = value.decode()
    candidate = values.get("CONTAINER_ID") or values.get("VAST_CONTAINERLABEL")
    if candidate and re.fullmatch(r"[A-Za-z0-9._-]{1,80}", candidate):
        instance_id = candidate
except (OSError, UnicodeDecodeError):
    pass

heartbeat = {
    "schema_version": 1,
    "instance_id": instance_id,
    "commit_sha": os.environ["SOURCE_COMMIT"],
    "milestone": os.environ["MILESTONE"],
    "phase": os.environ["PHASE"],
    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "failure_reason": os.environ["FAILURE"],
    "redacted": True,
}
Path(os.environ["HEARTBEAT_FILE"]).write_text(json.dumps(heartbeat, indent=2, sort_keys=True) + "\n")
PY

ssh_command='ssh -i /root/.ssh/ltx_csharp_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
if [[ ! -f /root/.ssh/ltx_csharp_deploy ]]; then
    echo "ephemeral deploy key is missing" >&2
    exit 1
fi

blob=$(git -C "$repo_root" hash-object -w "$heartbeat_file")
tree=$(printf '100644 blob %s\theartbeat.json\n' "$blob" | git -C "$repo_root" mktree)
parent=$(GIT_SSH_COMMAND="$ssh_command" git -C "$repo_root" ls-remote origin refs/heads/codex/ltx-csharp-m7a-status | awk '{print $1}')
commit_args=("$tree")
if [[ -n $parent ]]; then
    commit_args+=("-p" "$parent")
fi
status_commit=$(
    printf 'status: %s %s\n' "$milestone" "$phase" |
        GIT_AUTHOR_NAME='LTX remote heartbeat' \
        GIT_AUTHOR_EMAIL='heartbeat@localhost' \
        GIT_COMMITTER_NAME='LTX remote heartbeat' \
        GIT_COMMITTER_EMAIL='heartbeat@localhost' \
        git -C "$repo_root" commit-tree "${commit_args[@]}"
)
GIT_SSH_COMMAND="$ssh_command" git -C "$repo_root" push --force origin \
    "$status_commit:refs/heads/codex/ltx-csharp-m7a-status" >/dev/null
echo "redacted heartbeat published"
