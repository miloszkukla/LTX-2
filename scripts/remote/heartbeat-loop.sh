#!/usr/bin/env bash
set -Eeuo pipefail

milestone=${1:?usage: heartbeat-loop.sh MILESTONE PHASE}
phase=${2:?usage: heartbeat-loop.sh MILESTONE PHASE}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

while true; do
    "$script_dir/publish-heartbeat.sh" "$milestone" "$phase" none
    sleep 240
done
