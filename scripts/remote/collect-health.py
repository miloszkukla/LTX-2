#!/usr/bin/env python3
"""Collect a single redacted host health sample."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/health.json")
    args = parser.parse_args()
    values = [
        value.strip()
        for value in command(
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gpu": {
            "memory_used_mib": int(values[0]),
            "memory_total_mib": int(values[1]),
            "temperature_c": int(values[2]),
            "power_w": float(values[3]),
            "utilization_percent": int(values[4]),
        },
        "disk_free_bytes": int(command("df", "-PB1", str(ROOT)).splitlines()[1].split()[3]),
        "redacted": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("health sample collected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
