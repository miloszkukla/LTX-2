#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/root/.local/bin:$PATH"

milestone=${1:-}
if [[ ! $milestone =~ ^M[0-7]$ ]]; then
    echo "usage: $0 <M0..M7>" >&2
    exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $milestone == M6 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M6 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_m1_m4_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    expected_m5_m6_plan_sha=c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M2.md | awk '{print $1}')
    actual_m3_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M3.md | awk '{print $1}')
    actual_m4_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M4.md | awk '{print $1}')
    actual_m5_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M5.md | awk '{print $1}')
    actual_m6_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m2_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m3_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m4_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m5_plan_sha != "$expected_m5_m6_plan_sha" || \
          $actual_m6_plan_sha != "$expected_m5_m6_plan_sha" ]]; then
        echo "M0/M1/M2/M3/M4/M5/M6 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M5/summary.json").read_text())
if summary.get("milestone") != "M5" or summary.get("state") != "accepted":
    raise SystemExit("M5 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M6"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    scripts/remote/run-m3-core.sh "$artifact_dir/core-model.json"
    scripts/remote/run-m4-cuda.sh "$artifact_dir/cuda-quantization-lora.json"
    scripts/remote/run-m5-media.sh "$artifact_dir/media-pipelines.json"
    scripts/remote/run-m6-training.sh "$artifact_dir/low-vram-training.json"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M6")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
m5 = json.loads((root / "media-pipelines.json").read_text())
m6 = json.loads((root / "low-vram-training.json").read_text())
if any(result.get("result") != "pass" for result in (smoke, foundation, storage, core, m4, m5, m6)):
    raise SystemExit("M6 result artifact is not passing")
expected_totals = [
    (foundation, {"passed": 2, "failed": 0, "skipped": 0}),
    (storage, {"passed": 16, "failed": 0, "skipped": 0}),
    (core, {"passed": 34, "failed": 0, "skipped": 0}),
    (m4, {"passed": 45, "failed": 0, "skipped": 0}),
    (m5, {"passed": 51, "failed": 0, "skipped": 0}),
    (m6, {"passed": 21, "failed": 0, "skipped": 0}),
]
if any(result.get("test_totals") != expected for result, expected in expected_totals):
    raise SystemExit("M6 prerequisite or trainer totals mismatch")
if m6.get("suite_totals") != {"preprocessing": 5, "training": 11, "low_vram": 5}:
    raise SystemExit("M6 required suite totals mismatch")
if m6.get("fixture_revision") != "m6-low-vram-training-v1":
    raise SystemExit("M6 fixture revision mismatch")
if m6.get("numerical_tolerances") != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"},
    "bf16": {"rtol": 2e-2, "atol": 5e-3, "result": "pass"},
}:
    raise SystemExit("M6 numerical tolerance mismatch")
training = m6.get("training", {})
if training.get("mode") != "single_gpu_lora_one_step" or \
        training.get("lora_a_gradient") != "deterministic_bf16_pass" or \
        training.get("lora_b_gradient") != "deterministic_bf16_pass":
    raise SystemExit("M6 deterministic loss/gradient gate mismatch")
profile = m6.get("low_vram_profile", {})
required_profile = {
    "batch_size": 1,
    "mixed_precision": "bf16",
    "base_quantization": "int8-rowwise",
    "optimizer": "cpu-offloaded-adamw8bit",
    "gradient_checkpointing": True,
    "base_weight_residence": "cpu_int8",
    "optimizer_state_residence": "cpu_int8",
    "visible_cuda_devices": 1,
}
if any(profile.get(key) != value for key, value in required_profile.items()):
    raise SystemExit("M6 low-VRAM profile gate mismatch")
if profile.get("gpu_peak_memory_mib", 0) >= profile.get("gpu_total_memory_mib", 0):
    raise SystemExit("M6 GPU memory gate failed")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M6")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
m5 = json.loads((root / "media-pipelines.json").read_text())
m6 = json.loads((root / "low-vram-training.json").read_text())
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M6",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M6",
    "plan_sha256": "c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2",
    "verification_command": "scripts/remote/verify-milestone.sh M6",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m2-storage.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m3-core.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m4-cuda.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m5-media.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m6-training.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": {**preflight["gpu"], **m6["low_vram_profile"]},
    "abi": smoke["abi"],
    "test_totals": {"passed": 173, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "cuda_quantization_lora_regression": "45_pass",
        "media_pipeline_regression": "51_pass",
        "preprocessing": "5_pass",
        "one_step_lora_training": "11_pass",
        "low_vram_profile": "5_pass",
    },
    "suite_totals": m6["suite_totals"],
    "preprocessing": m6["preprocessing"],
    "training": m6["training"],
    "low_vram_profile": m6["low_vram_profile"],
    "fixture_revision": m6["fixture_revision"],
    "fixture_set_sha256": m6["fixture_set_sha256"],
    "fixture_sha256": m6["fixture_sha256"],
    "dependencies": {
        **storage["dependencies"],
        **core["dependencies"],
        **m4["dependencies"],
        **m5["dependencies"],
        **m6["dependencies"],
    },
    "numerical_tolerances": m6["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M6 verified none
    echo "M6 accepted"
    exit 0
fi

if [[ $milestone == M5 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M5 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_accepted_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    expected_m5_plan_sha=c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M2.md | awk '{print $1}')
    actual_m3_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M3.md | awk '{print $1}')
    actual_m4_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M4.md | awk '{print $1}')
    actual_m5_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_accepted_plan_sha" || \
          $actual_m2_plan_sha != "$expected_accepted_plan_sha" || \
          $actual_m3_plan_sha != "$expected_accepted_plan_sha" || \
          $actual_m4_plan_sha != "$expected_accepted_plan_sha" || \
          $actual_m5_plan_sha != "$expected_m5_plan_sha" ]]; then
        echo "M0/M1/M2/M3/M4/M5 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M4/summary.json").read_text())
if summary.get("milestone") != "M4" or summary.get("state") != "accepted":
    raise SystemExit("M4 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M5"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    scripts/remote/run-m3-core.sh "$artifact_dir/core-model.json"
    scripts/remote/run-m4-cuda.sh "$artifact_dir/cuda-quantization-lora.json"
    scripts/remote/run-m5-media.sh "$artifact_dir/media-pipelines.json"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M5")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
m5 = json.loads((root / "media-pipelines.json").read_text())
if any(result.get("result") != "pass" for result in (smoke, foundation, storage, core, m4, m5)):
    raise SystemExit("M5 result artifact is not passing")
if foundation.get("test_totals") != {"passed": 2, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 foundation regression totals mismatch")
if storage.get("test_totals") != {"passed": 16, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 storage regression totals mismatch")
if core.get("test_totals") != {"passed": 34, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 core-model regression totals mismatch")
if m4.get("test_totals") != {"passed": 45, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 CUDA/quantization/LoRA regression totals mismatch")
if m5.get("test_totals") != {"passed": 51, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 media/pipeline totals mismatch")
if m5.get("suite_totals") != {
    "media": 9,
    "pipelines": 12,
    "cli_modes": 12,
    "cli_help": 12,
    "cli_errors": 6,
}:
    raise SystemExit("M5 required suite totals mismatch")
inventory = m5.get("pipeline_inventory", {})
if inventory.get("total") != 15 or inventory.get("in_scope") != 12 or inventory.get("tested") != 12:
    raise SystemExit("M5 does not test every in-scope single-GPU pipeline")
deferred = inventory.get("deferred", [])
if len(deferred) != 3 or any("_mgpu" not in item.get("module", "") for item in deferred):
    raise SystemExit("M5 deferred pipeline inventory mismatch")
if m5.get("fixture_revision") != "m5-media-pipelines-v1":
    raise SystemExit("M5 fixture revision mismatch")
if m5.get("numerical_tolerances") != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"},
}:
    raise SystemExit("M5 numerical tolerance mismatch")
cli = m5.get("cli_behavior", {})
if len(cli.get("mode_success", {})) != 12 or len(cli.get("help_behavior", {})) != 12 or \
        len(cli.get("error_behavior", {})) != 6:
    raise SystemExit("M5 CLI exit/output behavior coverage mismatch")
media = m5.get("media", {})
if media.get("hdr_ic_lora_output") != "nine_exr_frames_and_prores_mov_pass" or \
        media.get("exr") != "openimageio_native_read_write_sequence_pass":
    raise SystemExit("M5 HDR/EXR output behavior mismatch")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M5")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
m5 = json.loads((root / "media-pipelines.json").read_text())
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M5",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M5",
    "plan_sha256": "c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2",
    "verification_command": "scripts/remote/verify-milestone.sh M5",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m2-storage.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m3-core.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m4-cuda.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m5-media.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": preflight["gpu"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 152, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "cuda_quantization_lora_regression": "45_pass",
        "media": "9_pass",
        "single_gpu_pipelines": "12_pass",
        "cli_success_help_errors": "30_pass",
    },
    "suite_totals": m5["suite_totals"],
    "pipeline_inventory": m5["pipeline_inventory"],
    "cli_behavior": m5["cli_behavior"],
    "media": m5["media"],
    "fixture_revision": m5["fixture_revision"],
    "fixture_set_sha256": m5["fixture_set_sha256"],
    "fixture_sha256": m5["fixture_sha256"],
    "dependencies": {**storage["dependencies"], **core["dependencies"], **m4["dependencies"], **m5["dependencies"]},
    "numerical_tolerances": m5["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M5 verified none
    echo "M5 accepted"
    exit 0
fi

if [[ $milestone == M4 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M4 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_current_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M2.md | awk '{print $1}')
    actual_m3_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M3.md | awk '{print $1}')
    actual_m4_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_current_plan_sha" || \
          $actual_m2_plan_sha != "$expected_current_plan_sha" || \
          $actual_m3_plan_sha != "$expected_current_plan_sha" || \
          $actual_m4_plan_sha != "$expected_current_plan_sha" ]]; then
        echo "M0/M1/M2/M3/M4 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M3/summary.json").read_text())
if summary.get("milestone") != "M3" or summary.get("state") != "accepted":
    raise SystemExit("M3 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M4"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    scripts/remote/run-m3-core.sh "$artifact_dir/core-model.json"
    scripts/remote/run-m4-cuda.sh "$artifact_dir/cuda-quantization-lora.json"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M4")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
if any(result.get("result") != "pass" for result in (smoke, foundation, storage, core, m4)):
    raise SystemExit("M4 result artifact is not passing")
if foundation.get("test_totals") != {"passed": 2, "failed": 0, "skipped": 0}:
    raise SystemExit("M4 foundation regression totals mismatch")
if storage.get("test_totals") != {"passed": 16, "failed": 0, "skipped": 0}:
    raise SystemExit("M4 storage regression totals mismatch")
if core.get("test_totals") != {"passed": 34, "failed": 0, "skipped": 0}:
    raise SystemExit("M4 core-model regression totals mismatch")
if m4.get("test_totals") != {"passed": 45, "failed": 0, "skipped": 0}:
    raise SystemExit("M4 CUDA/quantization/LoRA totals mismatch")
if m4.get("suite_totals") != {
    "native_kernels": 24,
    "native_bindings": 15,
    "quantization": 4,
    "lora": 2,
}:
    raise SystemExit("M4 required suite totals mismatch")
kernel_inventory = m4.get("kernel_inventory", {})
if kernel_inventory.get("in_scope") != 24 or kernel_inventory.get("tested") != 24:
    raise SystemExit("M4 does not test every in-scope native kernel")
if len(kernel_inventory.get("deferred", [])) != 5:
    raise SystemExit("M4 deferred kernel inventory mismatch")
binding_inventory = m4.get("native_binding_inventory", {})
if binding_inventory.get("in_scope") != 15 or binding_inventory.get("tested") != 15:
    raise SystemExit("M4 does not test every in-scope native binding")
if m4.get("fixture_revision") != "m4-cuda-quantization-lora-v1":
    raise SystemExit("M4 fixture revision mismatch")
if m4.get("numerical_tolerances") != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"},
    "bf16_fp8": {"rtol": 2e-2, "atol": 5e-3, "result": "pass"},
}:
    raise SystemExit("M4 numerical tolerance mismatch")
if m4.get("lora") != {
    "nvfp4_fuse": "pass",
    "nvfp4_unfuse": "exact_original_snapshot_pass",
    "bf16_regression": "covered_by_M2_storage",
}:
    raise SystemExit("M4 LoRA fuse/unfuse gate mismatch")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M4")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
m4 = json.loads((root / "cuda-quantization-lora.json").read_text())
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M4",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M4",
    "plan_sha256": "7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea",
    "verification_command": "scripts/remote/verify-milestone.sh M4",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m2-storage.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m3-core.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m4-cuda.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": preflight["gpu"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 101, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "native_kernel_correctness": "24_pass",
        "native_bindings": "15_pass",
        "quantization": "4_pass",
        "lora_fuse_unfuse": "2_pass",
    },
    "suite_totals": m4["suite_totals"],
    "kernel_inventory": m4["kernel_inventory"],
    "native_binding_inventory": m4["native_binding_inventory"],
    "quantization": m4["quantization"],
    "lora": m4["lora"],
    "fixture_revision": m4["fixture_revision"],
    "fixture_set_sha256": m4["fixture_set_sha256"],
    "fixture_sha256": m4["fixture_sha256"],
    "dependencies": {**storage["dependencies"], **core["dependencies"], **m4["dependencies"]},
    "numerical_tolerances": m4["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M4 verified none
    echo "M4 accepted"
    exit 0
fi

if [[ $milestone == M3 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M3 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_current_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M2.md | awk '{print $1}')
    actual_m3_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_current_plan_sha" || \
          $actual_m2_plan_sha != "$expected_current_plan_sha" || \
          $actual_m3_plan_sha != "$expected_current_plan_sha" ]]; then
        echo "M0/M1/M2/M3 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M2/summary.json").read_text())
if summary.get("milestone") != "M2" or summary.get("state") != "accepted":
    raise SystemExit("M2 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M3"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    scripts/remote/run-m3-core.sh "$artifact_dir/core-model.json"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M3")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
if any(result.get("result") != "pass" for result in (smoke, foundation, storage, core)):
    raise SystemExit("M3 result artifact is not passing")
if foundation.get("test_totals") != {"passed": 2, "failed": 0, "skipped": 0}:
    raise SystemExit("M3 foundation regression totals mismatch")
if storage.get("test_totals") != {"passed": 16, "failed": 0, "skipped": 0}:
    raise SystemExit("M3 storage regression totals mismatch")
if core.get("test_totals") != {"passed": 34, "failed": 0, "skipped": 0}:
    raise SystemExit("M3 core-model totals mismatch")
if core.get("suite_totals") != {
    "transformer": 7,
    "vae": 8,
    "audio_vocoder": 9,
    "conditioning": 4,
    "offload": 6,
}:
    raise SystemExit("M3 required suite totals mismatch")
if core.get("fixture_revision") != "m3-core-model-execution-v1":
    raise SystemExit("M3 fixture revision mismatch")
if core.get("numerical_tolerances") != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"},
    "bf16": {"rtol": 2e-2, "atol": 5e-3, "result": "pass"},
}:
    raise SystemExit("M3 numerical tolerance mismatch")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M3")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
core = json.loads((root / "core-model.json").read_text())
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M3",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M3",
    "plan_sha256": "7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea",
    "verification_command": "scripts/remote/verify-milestone.sh M3",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m2-storage.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m3-core.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": preflight["gpu"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 56, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        **core["tests"],
    },
    "suite_totals": core["suite_totals"],
    "fixture_revision": core["fixture_revision"],
    "fixture_set_sha256": core["fixture_set_sha256"],
    "fixture_sha256": core["fixture_sha256"],
    "dependencies": {**storage["dependencies"], **core["dependencies"]},
    "numerical_tolerances": core["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M3 verified none
    echo "M3 accepted"
    exit 0
fi

if [[ $milestone == M2 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M2 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_m1_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    expected_m2_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_m1_plan_sha" || \
          $actual_m2_plan_sha != "$expected_m2_plan_sha" ]]; then
        echo "M0/M1/M2 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M1/summary.json").read_text())
if summary.get("milestone") != "M1" or summary.get("state") != "accepted":
    raise SystemExit("M1 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M2"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M2")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
if smoke.get("result") != "pass" or foundation.get("result") != "pass" or storage.get("result") != "pass":
    raise SystemExit("M2 result artifact is not passing")
if foundation.get("test_totals") != {"passed": 2, "failed": 0, "skipped": 0}:
    raise SystemExit("M2 foundation regression totals mismatch")
if storage.get("test_totals") != {"passed": 16, "failed": 0, "skipped": 0}:
    raise SystemExit("M2 storage totals mismatch")
if storage.get("fixture_revision") != "m2-storage-model-loading-v1":
    raise SystemExit("M2 fixture revision mismatch")
if storage.get("numerical_tolerances") != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"},
    "bf16": {"rtol": 2e-2, "atol": 5e-3, "result": "pass"},
}:
    raise SystemExit("M2 numerical tolerance mismatch")
if storage.get("round_trip") != {
    "safetensors": "exact_dtype_shape_payload_metadata",
    "lora": "exact_dtype_shape_payload_metadata",
}:
    raise SystemExit("M2 round-trip gate mismatch")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M2")
smoke = json.loads((root / "abi-smoke.json").read_text())
foundation = json.loads((root / "foundation-parity.json").read_text())
storage = json.loads((root / "storage.json").read_text())
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M2",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M2",
    "plan_sha256": "7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea",
    "verification_command": "scripts/remote/verify-milestone.sh M2",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m2-storage.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": preflight["gpu"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 22, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        **storage["tests"],
    },
    "fixture_revision": storage["fixture_revision"],
    "fixture_set_sha256": storage["fixture_set_sha256"],
    "fixture_sha256": storage["fixture_sha256"],
    "dependencies": storage["dependencies"],
    "round_trip": storage["round_trip"],
    "numerical_tolerances": storage["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M2 verified none
    echo "M2 accepted"
    exit 0
fi

if [[ $milestone == M1 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M1 blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_m1_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || $actual_m1_plan_sha != "$expected_m1_plan_sha" ]]; then
        echo "M0/M1 plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M0/summary.json").read_text())
if summary.get("milestone") != "M0" or summary.get("state") != "accepted":
    raise SystemExit("M0 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M1"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/parity.json"
    git diff --check

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M1")
smoke = json.loads((root / "abi-smoke.json").read_text())
parity = json.loads((root / "parity.json").read_text())
if smoke.get("result") != "pass" or parity.get("result") != "pass":
    raise SystemExit("M1 result artifact is not passing")
if smoke["abi"] != {
    "cuda": "13.2",
    "cxx11_abi": 1,
    "ltx": "1.0",
    "pytorch": "2.13.0",
    "torchsharp_managed": "0.107.0",
    "torchsharp_revision": "8f4def03b641b6753f18076aa5438f8eaaef2d30",
}:
    raise SystemExit("M1 ABI artifact mismatch")
if parity["test_totals"] != {"passed": 2, "failed": 0, "skipped": 0}:
    raise SystemExit("M1 parity totals mismatch")
if parity["numerical_tolerances"] != {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "result": "pass"}
}:
    raise SystemExit("M1 parity tolerance mismatch")

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
m0_preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())
summary = {
    "schema_version": 1,
    "milestone": "M1",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "plan_revision": "M1",
    "plan_sha256": "7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea",
    "verification_command": "scripts/remote/verify-milestone.sh M1",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        {"command": "scripts/remote/run-m1-smoke.sh", "exit_code": 0},
        {"command": "scripts/remote/run-m1-parity.sh", "exit_code": 0},
    ],
    "toolchain": m0_preflight["toolchain"],
    "gpu": m0_preflight["gpu"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 6, "failed": 0, "skipped": 0},
    "tests": {**smoke["tests"], "parity_fixtures": "2_pass"},
    "fixture_revision": parity["fixture_revision"],
    "fixture_sha256": parity["fixture_sha256"],
    "numerical_tolerances": parity["numerical_tolerances"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    scripts/remote/publish-heartbeat.sh M1 verified none
    echo "M1 accepted"
    exit 0
fi

if [[ $milestone != M0 ]]; then
    echo "$milestone has not started; M0 through M3 are the implemented acceptance gates" >&2
    exit 2
fi

artifact_dir="$repo_root/artifacts/M0"
mkdir -p "$artifact_dir"

expected_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
actual_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
if [[ $actual_plan_sha != $expected_plan_sha ]]; then
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
