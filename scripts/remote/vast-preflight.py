#!/usr/bin/env python3
"""Run sanitized Vast CLI search/status checks and enforce the M0 price/tier policy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def pid1_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        items = Path("/proc/1/environ").read_bytes().split(b"\0")
    except OSError:
        return result
    for item in items:
        if b"=" not in item:
            continue
        name, value = item.split(b"=", 1)
        try:
            result[name.decode()] = value.decode()
        except UnicodeDecodeError:
            continue
    return result


def vast_json(*args: str) -> object:
    completed = subprocess.run(
        ["vastai", *args, "--raw"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"vastai {' '.join(args)} did not return JSON") from error


def records(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("offers", "instances", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def numeric_id(value: str) -> int | None:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/M0/vast-preflight.json")
    args = parser.parse_args()

    environment = pid1_environment()
    current_id = numeric_id(environment.get("CONTAINER_ID", "") or environment.get("VAST_CONTAINERLABEL", ""))
    instances = records(vast_json("show", "instances"))
    current = next((item for item in instances if int(item.get("id", -1)) == current_id), None)
    if current is None:
        raise SystemExit("current Vast instance was not returned by authenticated `vastai show instances --raw`")

    current_price = float(current.get("dph_total", current.get("dph_base", 999.0)))
    gpu_name = str(current.get("gpu_name", ""))
    num_gpus = int(current.get("num_gpus", 0))
    direct_ssh = bool(current.get("ssh_host")) and bool(current.get("ssh_port"))
    if current_price > 0.60:
        raise SystemExit(f"routine worker hourly cap exceeded: {current_price:.4f} > 0.60")
    if num_gpus != 1 or "5090" not in gpu_name or not direct_ssh:
        raise SystemExit("current worker is not the required direct-SSH single RTX 5090 tier")

    routine = records(
        vast_json(
            "search",
            "offers",
            "gpu_name=RTX_5090 num_gpus=1 dph_total<=0.60 reliability>=0.95",
        )
    )
    h100 = records(
        vast_json(
            "search",
            "offers",
            "gpu_name in [H100_SXM,H100_PCIE] num_gpus=1 dph_total<=1.20 reliability>=0.95",
        )
    )

    def minimum_price(items: list[dict[str, object]]) -> float | None:
        prices = [float(item["dph_total"]) for item in items if item.get("dph_total") is not None]
        return min(prices) if prices else None

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cli": {"show_instances_raw": "pass", "search_offers_raw": "pass"},
        "authorization": {
            "routine_worker_max_usd_per_hour": 0.60,
            "h100_validation_worker_max_usd_per_hour": 1.20,
            "per_instance_duration": "unlimited",
            "total_spend": "unlimited",
        },
        "current_worker": {
            "instance_id": current_id,
            "gpu_name": gpu_name,
            "num_gpus": num_gpus,
            "dph_total_usd": current_price,
            "direct_ssh": direct_ssh,
            "within_cap": True,
        },
        "live_search": {
            "routine_compatible_offer_count": len(routine),
            "routine_min_dph_total_usd": minimum_price(routine),
            "h100_compatible_offer_count": len(h100),
            "h100_min_dph_total_usd": minimum_price(h100),
        },
        "redaction": "No account environment, host address, SSH port, or credential value is persisted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Vast CLI raw search/status passed; current RTX 5090 worker is within the $0.60/hour cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
