#!/usr/bin/env bash
set -Eeuo pipefail

milestone=${1:?usage: start-heartbeat.sh MILESTONE PHASE}
phase=${2:?usage: start-heartbeat.sh MILESTONE PHASE}
if [[ ! $milestone =~ ^M[0-7]$ || ! $phase =~ ^[A-Za-z0-9_.:-]{1,80}$ ]]; then
    echo "heartbeat fields must be short redacted identifiers" >&2
    exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
session=ltx-csharp-m7a-heartbeat

if tmux has-session -t "$session" 2>/dev/null; then
    echo "heartbeat tmux session already running"
    exit 0
fi
tmux new-session -d -s "$session" \
    "$script_dir/heartbeat-loop.sh '$milestone' '$phase'"
echo "heartbeat tmux session started"
