#!/usr/bin/env bash
set -Eeuo pipefail

output=${1:?usage: sample-gpu.sh OUTPUT.jsonl [INTERVAL_SECONDS]}
interval=${2:-30}
if [[ ! $interval =~ ^[1-9][0-9]*$ ]]; then
    echo "interval must be a positive integer" >&2
    exit 2
fi
mkdir -p "$(dirname "$output")"

while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    metrics=$(nvidia-smi \
        --query-gpu=memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader,nounits)
    IFS=',' read -r memory_used memory_total temperature power utilization <<<"$metrics"
    jq -cn \
        --arg timestamp "$timestamp" \
        --argjson memory_used "${memory_used// /}" \
        --argjson memory_total "${memory_total// /}" \
        --argjson temperature "${temperature// /}" \
        --argjson power "${power// /}" \
        --argjson utilization "${utilization// /}" \
        '{timestamp_utc:$timestamp,memory_used_mib:$memory_used,memory_total_mib:$memory_total,temperature_c:$temperature,power_w:$power,utilization_percent:$utilization}' \
        >>"$output"
    sleep "$interval"
done
