#!/usr/bin/env bash
set -Eeuo pipefail

milestone=${1:?usage: start-heartbeat.sh MILESTONE PHASE [FAILURE_CODE]}
phase=${2:?usage: start-heartbeat.sh MILESTONE PHASE [FAILURE_CODE]}
failure=${3:-none}
if [[ ! $milestone =~ ^M([0-6]|7A)$ || ! $phase =~ ^[A-Za-z0-9_.:-]{1,80}$ || ! $failure =~ ^[A-Za-z0-9_.:-]{1,160}$ ]]; then
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
    "$script_dir/heartbeat-loop.sh '$milestone' '$phase' '$failure'"
echo "heartbeat tmux session started"
