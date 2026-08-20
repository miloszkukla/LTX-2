#!/usr/bin/env python3
"""Verify a regenerated CUDA oracle without requiring device-specific byte identity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_FP32 = {"rtol": 1e-4, "atol": 1e-5}


def compare(accepted: Any, generated: Any, path: tuple[str | int, ...], *, allow_device: bool) -> None:
    label = ".".join(map(str, path)) or "root"
    if isinstance(accepted, dict):
        if not isinstance(generated, dict) or accepted.keys() != generated.keys():
            raise SystemExit(f"fixture object keys differ at {label}")
        for key in accepted:
            compare(accepted[key], generated[key], (*path, key), allow_device=allow_device)
        return
    if isinstance(accepted, list):
        if not isinstance(generated, list) or len(accepted) != len(generated):
            raise SystemExit(f"fixture array shape differs at {label}")
        for index, (left, right) in enumerate(zip(accepted, generated, strict=True)):
            compare(left, right, (*path, index), allow_device=allow_device)
        return
    if allow_device and path == ("oracle", "device"):
        if not isinstance(accepted, str) or not isinstance(generated, str) or not generated:
            raise SystemExit("fixture oracle device metadata is invalid")
        return
    if isinstance(accepted, float):
        if not isinstance(generated, (int, float)) or not math.isclose(
            accepted,
            generated,
            rel_tol=EXPECTED_FP32["rtol"],
            abs_tol=EXPECTED_FP32["atol"],
        ):
            raise SystemExit(f"fixture FP32 tolerance failed at {label}: {accepted!r} != {generated!r}")
        return
    if accepted != generated:
        raise SystemExit(f"fixture value differs at {label}: {accepted!r} != {generated!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--allow-device-name", action="store_true")
    args = parser.parse_args()

    accepted = json.loads(args.accepted.read_text())
    generated = json.loads(args.generated.read_text())
    if accepted.get("tolerances", {}).get("fp32") != EXPECTED_FP32:
        raise SystemExit("accepted fixture does not retain the required FP32 tolerance")
    if generated.get("tolerances", {}).get("fp32") != EXPECTED_FP32:
        raise SystemExit("regenerated fixture does not retain the required FP32 tolerance")
    compare(accepted, generated, (), allow_device=args.allow_device_name)
    print("cross-GPU fixture reproduction passed at FP32 rtol=1e-4, atol=1e-5")


if __name__ == "__main__":
    main()
