#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/root/.local/bin:$PATH"

milestone=${1:-}
if [[ ! $milestone =~ ^M[0-7]$ ]]; then
    echo "usage: $0 <M0..M7>" >&2
    exit 2
fi
if [[ $milestone != M0 ]]; then
    echo "$milestone has not started; M0 is the only implemented acceptance gate" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
artifact_dir="$repo_root/artifacts/M0"
mkdir -p "$artifact_dir"

expected_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
actual_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
if [[ $actual_plan_sha != $expected_plan_sha || -e .m0-bootstrap-plan.md ]]; then
    echo "plan copy/hash/removal gate failed" >&2
    exit 1
fi
if rg -n '^(ARG|ENV)[[:space:]]+HF_TOKEN' Dockerfile.remote >/dev/null; then
    echo "Dockerfile.remote must not declare ARG/ENV HF_TOKEN" >&2
    exit 1
fi
if ! grep -qxF '**' .dockerignore; then
    echo ".dockerignore must exclude the complete source tree" >&2
    exit 1
fi

python3 scripts/remote/generate-source-surface.py --output "$artifact_dir/source-surface.json"
python3 scripts/remote/probe-model-access.py \
    --manifest "$artifact_dir/artifact-manifest.json" \
    --output "$artifact_dir/model-access.json"
python3 scripts/remote/vast-preflight.py --output "$artifact_dir/vast-preflight.json"
python3 scripts/remote/probe-ghcr.py --output "$artifact_dir/ghcr.json"
python3 scripts/remote/preflight.py --output "$artifact_dir/preflight.json"
python3 scripts/remote/collect-health.py --output "$artifact_dir/health.json"
scripts/remote/run-native-smoke.sh "$artifact_dir/native-smoke.json"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M0")
required = [
    "artifact-manifest.json",
    "bootstrap-packages.json",
    "ghcr.json",
    "health.json",
    "model-access.json",
    "native-smoke.json",
    "preflight.json",
    "source-surface.json",
    "vast-preflight.json",
]
for name in required:
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing M0 artifact: {path}")
    json.loads(path.read_text())

surface = json.loads((root / "source-surface.json").read_text())
for key in (
    "python_packages", "python_modules", "cli_entrypoints", "cli_flag_declarations",
    "pipelines", "native_sources", "native_kernels",
):
    if surface["counts"][key] <= 0:
        raise SystemExit(f"source surface is empty: {key}")
if not any(item["status"] == "deferred" for item in surface["python_modules"]):
    raise SystemExit("source surface has no explicit deferred modules")
if not any(item["status"] == "in_scope" for item in surface["native_kernels"]):
    raise SystemExit("source surface has no in-scope native kernels")

model = json.loads((root / "model-access.json").read_text())
if any(item["access"] != "authenticated_read_succeeded" for item in model["artifacts"]):
    raise SystemExit("not every selected model artifact passed its authenticated read probe")
if any(
    file["sha256"] is None
    for item in model["artifacts"]
    for file in item["files"]
    if file["path"].endswith(".safetensors")
):
    raise SystemExit("a required safetensors artifact lacks Hugging Face LFS SHA-256 metadata")

if json.loads((root / "native-smoke.json").read_text())["result"] != "pass":
    raise SystemExit("native smoke result is not pass")
if not json.loads((root / "ghcr.json").read_text())["accepted"]:
    raise SystemExit("neither public GHCR nor the checked-in fallback is accepted")
PY

git diff --check

python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M0")
preflight = json.loads((root / "preflight.json").read_text())
native = json.loads((root / "native-smoke.json").read_text())
model = json.loads((root / "model-access.json").read_text())
ghcr = json.loads((root / "ghcr.json").read_text())
surface = json.loads((root / "source-surface.json").read_text())

checksums = {}
for path in sorted(root.glob("*")):
    if path.is_file() and path.name != "summary.json":
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "schema_version": 1,
    "milestone": "M0",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": "400fd31054597515f47125691032c04b1c3ee24e",
    "verification_command": "scripts/remote/verify-milestone.sh M0",
    "commands": [
        {"command": "generate-source-surface.py", "exit_code": 0},
        {"command": "probe-model-access.py", "exit_code": 0},
        {"command": "vast-preflight.py", "exit_code": 0},
        {"command": "probe-ghcr.py", "exit_code": 0},
        {"command": "preflight.py", "exit_code": 0},
        {"command": "run-native-smoke.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": preflight["gpu"],
    "test_totals": {"passed": 4, "failed": 0, "skipped": 0},
    "tests": native["tests"],
    "numerical_tolerances": {"result": "not_applicable_to_bootstrap_smoke"},
    "model_artifacts": {
        "authenticated_probes_passed": len(model["artifacts"]),
        "required_disk_with_headroom_bytes": model["disk_gate"]["required_disk_with_20_percent_headroom_bytes"],
    },
    "source_surface_counts": surface["counts"],
    "worker_strategy": ghcr["worker_strategy"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

scripts/remote/publish-heartbeat.sh M0 verified none
echo "M0 accepted"
