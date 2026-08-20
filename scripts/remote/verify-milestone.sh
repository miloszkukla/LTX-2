#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/root/.local/bin:$PATH"

milestone=${1:-}
if [[ ! $milestone =~ ^M([0-6]|7A|7B|7C|8)$ ]]; then
    echo "usage: $0 <M0..M6|M7A|M7B|M7C|M8>" >&2
    exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $milestone == M8 ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M8 blocked m8_acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_plan_sha=dbe05843ca8f2da269b0703469e222edfb72b55a9bbc824af0fa6bd882d8a157
    expected_m7b_source=9c705da5a58f37c4d1e75b19a0fa1a9e6a216597
    expected_m7a_source=51294b730a211b643118a4013f4b599d57a78de6
    [[ $(sha256sum /root/C_SHARP_PORT_PLAN.M8.md | awk '{print $1}') == "$expected_plan_sha" ]]
    [[ $(sha256sum C_SHARP_PORT_PLAN.md | awk '{print $1}') == "$expected_plan_sha" ]]
    [[ $(sha256sum C_SHARP_PORT_PLAN.M7A.md | awk '{print $1}') == 85da94394a9e37749d579bc333e79eedebe2a61f77324cacd7ecbd5d149247ba ]]
    [[ $(sha256sum C_SHARP_PORT_PLAN.M7B.md | awk '{print $1}') == c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2 ]]
    [[ $(git rev-parse HEAD^1) == "$expected_m7b_source" ]]
    [[ $(git rev-parse HEAD^2) == "$expected_m7a_source" ]]
    git merge-base --is-ancestor "$expected_m7b_source" HEAD
    git merge-base --is-ancestor "$expected_m7a_source" HEAD
    [[ -z $(git ls-files -u) ]]

    source_manifest_sha256() {
        python3 - <<'PY'
import hashlib
import subprocess
from pathlib import Path

excluded = ("artifacts/",)
paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
entries = []
for name in sorted(path for path in paths if path):
    if name == "PORT_STATUS.md" or name.startswith(excluded):
        continue
    payload = Path(name).read_bytes()
    entries.append(f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
print(hashlib.sha256("".join(entries).encode()).hexdigest())
PY
    }

    if [[ -f artifacts/M8/summary.json && -z $(git status --porcelain) ]]; then
        current_manifest=$(source_manifest_sha256)
        CURRENT_MANIFEST="$current_manifest" \
        EXPECTED_M7B_SOURCE="$expected_m7b_source" \
        EXPECTED_M7A_SOURCE="$expected_m7a_source" \
        python3 - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path("artifacts/M8/summary.json").read_text())
if summary.get("milestone") != "M8" or summary.get("state") != "accepted":
    raise SystemExit("M8 accepted summary is missing")
if summary.get("merge_parents") != {
    "codex_ltx_csharp_m7b": os.environ["EXPECTED_M7B_SOURCE"],
    "codex_ltx_csharp_m7a_m7c": os.environ["EXPECTED_M7A_SOURCE"],
}:
    raise SystemExit("M8 merge-parent evidence mismatch")
if summary.get("source_manifest_sha256") != os.environ["CURRENT_MANIFEST"]:
    raise SystemExit("M8 verified source-content manifest changed after verification")
if summary.get("test_totals") != {"passed": 267, "failed": 0, "skipped": 0}:
    raise SystemExit("M8 affected verification totals mismatch")
if summary.get("teardown_gate") != "closed_pending_explicit_user_review_authorization":
    raise SystemExit("M8 teardown gate was not retained")
PY
        git diff --check
        ssh_command='ssh -i /root/.ssh/ltx_csharp_m7a_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
        remote_tip=$(GIT_SSH_COMMAND="$ssh_command" git ls-remote origin refs/heads/codex/ltx-csharp | awk '{print $1}')
        [[ $remote_tip == "$(git rev-parse HEAD)" ]]
        trap - ERR
        scripts/remote/publish-heartbeat.sh M8 accepted none
        echo "M8 accepted"
        exit 0
    fi

    scripts/remote/verify-milestone.sh M7A
    .venv/bin/ruff check --ignore PLR0917 \
        scripts/remote/shard-safetensors.py \
        scripts/remote/record-m7b-delivery.py \
        scripts/remote/verify-m7b.py \
        packages/ltx-core/src/ltx_core/block_streaming/disk.py \
        packages/ltx-core/src/ltx_core/loader/helpers.py \
        packages/ltx-core/src/ltx_core/quantization/fp8_cast.py \
        packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py
    bash -n scripts/remote/run-m7b-sample.sh
    python3 -m py_compile \
        scripts/remote/shard-safetensors.py \
        scripts/remote/record-m7b-delivery.py \
        scripts/remote/verify-m7b.py \
        packages/ltx-core/src/ltx_core/block_streaming/disk.py \
        packages/ltx-core/src/ltx_core/loader/helpers.py \
        packages/ltx-core/src/ltx_core/quantization/fp8_cast.py \
        packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("artifacts/M7B")
summary = json.loads((root / "summary.json").read_text())
if summary.get("milestone") != "M7B" or summary.get("state") != "accepted":
    raise SystemExit("M8 accepted M7B prerequisite is missing")
if summary.get("test_totals") != {"passed": 14, "failed": 0, "skipped": 0}:
    raise SystemExit("M8 accepted M7B totals mismatch")
if summary.get("output", {}).get("sha256") != "3902bf2f534ca7409766e24c620921ecc22950bdc3b70fdc5c7a2e2a7c210ee9":
    raise SystemExit("M8 accepted M7B output identity mismatch")
for name, expected in summary.get("artifact_sha256", {}).items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"M8 M7B evidence checksum mismatch: {name}")
PY
    scripts/remote/verify-milestone.sh M7C
    git diff --check

    mkdir -p artifacts/M8
    verified_source_sha=$(git rev-parse HEAD)
    source_manifest=$(source_manifest_sha256)
    VERIFIED_SOURCE_SHA="$verified_source_sha" \
    SOURCE_MANIFEST="$source_manifest" \
    EXPECTED_PLAN_SHA="$expected_plan_sha" \
    EXPECTED_M7B_SOURCE="$expected_m7b_source" \
    EXPECTED_M7A_SOURCE="$expected_m7a_source" \
    python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

summaries = {
    name: json.loads(Path(f"artifacts/{name}/summary.json").read_text())
    for name in ("M7A", "M7B", "M7C")
}
expected_totals = {
    "M7A": {"passed": 192, "failed": 0, "skipped": 0},
    "M7B": {"passed": 14, "failed": 0, "skipped": 0},
    "M7C": {"passed": 66, "failed": 0, "skipped": 0},
}
for name, summary in summaries.items():
    if summary.get("milestone") != name or summary.get("state") != "accepted":
        raise SystemExit(f"M8 accepted prerequisite missing: {name}")
    if summary.get("test_totals") != expected_totals[name]:
        raise SystemExit(f"M8 accepted prerequisite totals mismatch: {name}")

videos = summaries["M7C"]["videos"]
expected_videos = {
    "csharp": "9bd4d680246737dcac62226b9c5c1984cb9f1d093bfec5bd0704fc952a1785af",
    "python": "16a15b7e68d94087da1ed1c3217e22dfacdaa5ceedda21ab83c5ff6b24ac32fd",
}
if any(videos[name]["sha256"] != checksum for name, checksum in expected_videos.items()):
    raise SystemExit("M8 paired M7C video identity mismatch")

artifact_sha256 = {
    f"{name}/summary.json": hashlib.sha256(Path(f"artifacts/{name}/summary.json").read_bytes()).hexdigest()
    for name in summaries
}
report = {
    "schema_version": 1,
    "milestone": "M8",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "verification_command": "scripts/remote/verify-milestone.sh M8",
    "verified_source_sha": os.environ["VERIFIED_SOURCE_SHA"],
    "source_manifest_sha256": os.environ["SOURCE_MANIFEST"],
    "merge_parents": {
        "codex_ltx_csharp_m7b": os.environ["EXPECTED_M7B_SOURCE"],
        "codex_ltx_csharp_m7a_m7c": os.environ["EXPECTED_M7A_SOURCE"],
    },
    "plan_revision": "M8",
    "plan_sha256": os.environ["EXPECTED_PLAN_SHA"],
    "commands": [
        {"command": "scripts/remote/verify-milestone.sh M7A", "exit_code": 0},
        {"command": ".venv/bin/ruff check <merged M7B Python sources>", "exit_code": 0},
        {"command": "bash -n scripts/remote/run-m7b-sample.sh", "exit_code": 0},
        {"command": "python3 -m py_compile <merged M7B Python sources>", "exit_code": 0},
        {"command": "validate accepted M7B redacted evidence checksums", "exit_code": 0},
        {"command": "scripts/remote/verify-milestone.sh M7C", "exit_code": 0},
        {"command": "git diff --check", "exit_code": 0},
    ],
    "suite_totals": {
        "M7A_full_regression": 192,
        "M7C_paired_video_regression": 66,
        "M7B_merged_source_checks": 4,
        "M8_integration_evidence_checks": 5,
    },
    "test_totals": {"passed": 267, "failed": 0, "skipped": 0},
    "accepted_prerequisites": {name: summaries[name]["source_reference_sha"] for name in summaries},
    "m7c_local_delivery_confirmation": {
        "basis": "explicit_coordinator_handoff_for_M8",
        "checksum_verified": True,
        "sha256": expected_videos,
    },
    "m7c_videos": videos,
    "artifact_sha256": artifact_sha256,
    "worktree_and_remote_tip_audit": "required_after_atomic_push_via_same_verification_command",
    "teardown_gate": "closed_pending_explicit_user_review_authorization",
    "destructive_actions_performed": False,
    "secrets_recorded": False,
}
Path("artifacts/M8/summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    echo "M8 affected verification passed; atomic checkpoint and read-only post-push audit remain"
    exit 0
fi

if [[ $milestone == M7C ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M7C blocked m7c_acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_plan_sha=6c7fbd1012661adba0dc7a9b89a8cbed7c1a1014fa9475b93531edad58578982
    expected_m7a_source=523d81b92c796048f2bbcc299c5c4f21e3fd5b5e
    [[ $(sha256sum /root/C_SHARP_PORT_PLAN.M7C.md | awk '{print $1}') == "$expected_plan_sha" ]]
    [[ $(git merge-base "$expected_m7a_source" HEAD) == "$expected_m7a_source" ]]
    "$repo_root/.venv/bin/python" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M7A/summary.json").read_text())
if summary.get("milestone") != "M7A" or summary.get("state") != "accepted":
    raise SystemExit("M7C accepted M7A prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M7C"
    mkdir -p "$artifact_dir"
    "$repo_root/.venv/bin/python" scripts/remote/collect-m7c-evidence.py --artifact-dir "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m5-media.sh "$artifact_dir/media-pipelines.json"

    checkpoint=/workspace/ltx-model-cache/M7A/ltx23-distilled-checkpoint/ltx-2.3-22b-distilled-1.1.safetensors
    upsampler=/workspace/ltx-model-cache/M7C/ltx23-spatial-upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    context="$repo_root/build/M7C/beach-volleyball-context.safetensors"
    stage_latents="$repo_root/build/M7C/diagnostics/python-stage-latents.safetensors"
    checkpoint_oracle="$repo_root/build/M7A/acceptance/real-checkpoint-oracle.safetensors"
    torchsharp_library="$repo_root/build/M1/torchsharp-build/LibTorchSharp/libLibTorchSharp.so"
    library_path=$(scripts/remote/m1-library-path.sh)
    scripts/remote/build-m5-media.sh >/dev/null
    export LD_LIBRARY_PATH="$library_path:$repo_root/build/M1/torchsharp-build/LibTorchSharp:$repo_root/build/M1/ltx-cuda:$repo_root/build/M5/ltx-media${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    [[ $(sha256sum "$checkpoint" | awk '{print $1}') == b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc ]]
    [[ $(sha256sum "$upsampler" | awk '{print $1}') == 5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed ]]
    [[ $(sha256sum "$context" | awk '{print $1}') == f4e7df3f9768932049c4ab3a2f733db0f39b9144e2ed6bfa46d01248f49fa495 ]]

    upsampler_output=$(dotnet run \
        --project tests/Ltx.UpsamplerTests/Ltx.UpsamplerTests.csproj \
        --configuration Release --no-build -- \
        "$checkpoint" "$upsampler" "$repo_root/tests/fixtures/M7C/upsampler.safetensors" "$torchsharp_library")
    [[ $upsampler_output == *"C# learned spatial upsampler: pass"* ]]

    transformer_output=$(dotnet run \
        --project tests/Ltx.VideoDecoderTests/Ltx.VideoDecoderTests.csproj \
        --configuration Release --no-build -- \
        transformer "$checkpoint" "$stage_latents" "$context" "$torchsharp_library" \
        "$artifact_dir/production-transformer-regression.json")
    [[ $transformer_output == *"M7C production transformer:"* ]]

    decoder_preview="$repo_root/build/M7C/diagnostics/csharp-decode-python-latent-regression.mp4"
    decoder_output=$(dotnet run \
        --project tests/Ltx.VideoDecoderTests/Ltx.VideoDecoderTests.csproj \
        --configuration Release --no-build -- \
        "$checkpoint" "$stage_latents" "$torchsharp_library" "$decoder_preview")
    [[ $decoder_output == *"M7C C# tiled decode: pass"* ]]

    runtime_output=$(dotnet run \
        --project tests/Ltx.CheckpointRuntimeTests/Ltx.CheckpointRuntimeTests.csproj \
        --configuration Release --no-build -- \
        "$checkpoint" "$checkpoint_oracle" "$torchsharp_library" "$artifact_dir/checkpoint-runtime.json")
    [[ $runtime_output == *"M7A real checkpoint runtime: 6 passed, 0 failed"* ]]

    csharp_decode_log="$artifact_dir/csharp-full-decode.stderr.log"
    python_decode_log="$artifact_dir/python-full-decode.stderr.log"
    ffmpeg -v error -i "$repo_root/build/M7C/videos/m7c-csharp-beach-volleyball.mp4" -f null - 2>"$csharp_decode_log"
    ffmpeg -v error -i "$repo_root/build/M7C/videos/m7c-python-beach-volleyball.mp4" -f null - 2>"$python_decode_log"
    [[ ! -s $csharp_decode_log && ! -s $python_decode_log ]]

    UPSAMPLER_OUTPUT="$upsampler_output" \
    TRANSFORMER_OUTPUT="$transformer_output" \
    DECODER_OUTPUT="$decoder_output" \
    RUNTIME_OUTPUT="$runtime_output" \
    DECODER_PREVIEW="$decoder_preview" \
    "$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

preview = Path(os.environ["DECODER_PREVIEW"])
probe = json.loads(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(preview)], text=True
))
video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
if video.get("width") != 1536 or video.get("height") != 1024 or video.get("nb_frames") != "1":
    raise SystemExit("M7C tiled decoder preview contract mismatch")
report = {
    "schema_version": 1,
    "result": "pass",
    "test_totals": {"passed": 66, "failed": 0, "skipped": 0},
    "suites": {
        "abi_smoke": 5,
        "media_pipelines": 52,
        "learned_spatial_upsampler": 1,
        "production_geometry_transformer": 1,
        "full_resolution_tiled_decoder": 1,
        "checkpoint_runtime": 6,
    },
    "output": {
        "upsampler": os.environ["UPSAMPLER_OUTPUT"],
        "transformer": os.environ["TRANSFORMER_OUTPUT"],
        "decoder": os.environ["DECODER_OUTPUT"],
        "runtime": os.environ["RUNTIME_OUTPUT"],
    },
    "decoder_preview": {
        "path": str(preview),
        "sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
        "ffprobe": probe,
    },
    "secrets_recorded": False,
}
Path("artifacts/M7C/tests.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

    git diff --check
    "$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M7C")
load = lambda name: json.loads((root / name).read_text())
reports = {name: load(name) for name in (
    "abi-smoke.json", "media-pipelines.json", "videos.json", "comparison.json", "latents.json",
    "models.json", "hardware.json", "production-transformer.json", "production-transformer-regression.json",
    "checkpoint-runtime.json", "port-repair.json", "tests.json",
)}
if any(report.get("result", "pass") != "pass" for report in reports.values()):
    raise SystemExit("M7C evidence or regression result is not passing")

videos = reports["videos.json"]
if videos.get("execution_order") != ["native_csharp_two_stage", "python_reference_two_stage"]:
    raise SystemExit("M7C C#-first execution order is missing")
if videos.get("prompt_sha256") != "fe9c3956eff10418967a912607b3c29ae77b9920a7013f62e692fa06742d855d":
    raise SystemExit("M7C exact prompt mismatch")
settings = videos.get("settings", {})
required_settings = {"seed": 20260821, "frames": 241, "frame_rate": 24, "width": 1536, "height": 1024}
if any(settings.get(key) != value for key, value in required_settings.items()) or \
        settings.get("stage1_sigmas") != [1, .99375, .9875, .98125, .975, .909375, .725, .421875, 0] or \
        settings.get("stage2_sigmas") != [.909375, .725, .421875, 0]:
    raise SystemExit("M7C sampling contract mismatch")
expected_hashes = {
    "csharp": "9bd4d680246737dcac62226b9c5c1984cb9f1d093bfec5bd0704fc952a1785af",
    "python": "16a15b7e68d94087da1ed1c3217e22dfacdaa5ceedda21ab83c5ff6b24ac32fd",
}
for name, expected in expected_hashes.items():
    item = videos["videos"][name]
    if item.get("sha256") != expected or item.get("full_stream_decode") != "pass":
        raise SystemExit(f"M7C {name} video identity/decode mismatch")

comparison = reports["comparison.json"]
if comparison["video_metrics"]["ssim"]["all"] < .80 or \
        comparison["video_metrics"]["psnr_db"]["average"] < 20 or \
        min(comparison["audio_apsnr_db"].values()) < 100:
    raise SystemExit("M7C paired-output comparison gate failed")
latents = reports["latents.json"]
if latents["comparisons"]["stage1_step0_video_input"]["exact_fraction"] != 1 or \
        latents["comparisons"]["stage1_step0_video_velocity"]["cosine"] < .999 or \
        not (.8 < latents["csharp"]["stage1_video"]["standard_deviation"] < 1.2) or \
        not (.8 < latents["csharp"]["stage2_video"]["standard_deviation"] < 1.2):
    raise SystemExit("M7C latent/content parity gate failed")

models = reports["models.json"]
if models["checkpoint"]["revision"] != "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce" or \
        models["text_encoder"]["revision"] != "68f7ee4fbd59087436ada77ed2d62f373fdd4482":
    raise SystemExit("M7C pinned model revision mismatch")
hardware = reports["hardware.json"]
gpu = hardware["gpu"]
if not gpu["name"].startswith(("NVIDIA A100", "NVIDIA H100")) or gpu["memory_total_mib"] < 80_000 or \
        max(gpu["csharp_peak_memory_used_mib"], gpu["python_peak_memory_used_mib"]) >= gpu["memory_total_mib"] or \
        hardware["offload"] != {"csharp": "required_disk_weight_streaming_after_resident_cuda_oom", "python": "none_resident"}:
    raise SystemExit("M7C GPU/offload gate failed")

for name in ("production-transformer.json", "production-transformer-regression.json"):
    transformer = reports[name]
    if transformer.get("execution") != "native_csharp_streamed_production_geometry_transformer" or \
            transformer.get("python_runtime_calls") != 0 or transformer["video"]["cosine"] < .999 or \
            transformer["audio"]["cosine"] < .999:
        raise SystemExit(f"M7C transformer parity gate failed: {name}")
runtime = reports["checkpoint-runtime.json"]
if runtime.get("test_totals") != {"passed": 6, "failed": 0, "skipped": 0} or \
        runtime.get("python_runtime_calls") != 0 or \
        runtime.get("comparisons", {}).get("convolutional_video_decoder_auto_tiled", {}).get("result") != "pass":
    raise SystemExit("M7C checkpoint runtime regression gate failed")
if reports["media-pipelines.json"].get("test_totals") != {"passed": 52, "failed": 0, "skipped": 0} or \
        reports["tests.json"].get("test_totals") != {"passed": 66, "failed": 0, "skipped": 0}:
    raise SystemExit("M7C regression totals mismatch")

checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
summary = {
    "schema_version": 1,
    "milestone": "M7C",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "accepted_m7a_source_sha": "523d81b92c796048f2bbcc299c5c4f21e3fd5b5e",
    "plan_revision": "M7C",
    "plan_sha256": "6c7fbd1012661adba0dc7a9b89a8cbed7c1a1014fa9475b93531edad58578982",
    "verification_command": "scripts/remote/verify-milestone.sh M7C",
    "execution_order": videos["execution_order"],
    "prompt_sha256": videos["prompt_sha256"],
    "settings": settings,
    "videos": videos["videos"],
    "comparison": comparison,
    "models": {key: models[key] for key in ("checkpoint", "spatial_upsampler", "text_encoder", "prompt_context")},
    "gpu": hardware["gpu"],
    "offload": hardware["offload"],
    "test_totals": {"passed": 66, "failed": 0, "skipped": 0},
    "port_repair": reports["port-repair.json"],
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    echo "M7C accepted"
    exit 0
fi

if [[ $milestone == M7B ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M7B blocked acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_m1_m4_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    expected_m5_m7b_plan_sha=c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2
    actual_m0_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}')
    actual_m1_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M1.md | awk '{print $1}')
    actual_m2_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M2.md | awk '{print $1}')
    actual_m3_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M3.md | awk '{print $1}')
    actual_m4_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M4.md | awk '{print $1}')
    actual_m5_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M5.md | awk '{print $1}')
    actual_m7b_plan_sha=$(sha256sum C_SHARP_PORT_PLAN.M7B.md | awk '{print $1}')
    if [[ $actual_m0_plan_sha != "$expected_m0_plan_sha" || \
          $actual_m1_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m2_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m3_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m4_plan_sha != "$expected_m1_m4_plan_sha" || \
          $actual_m5_plan_sha != "$expected_m5_m7b_plan_sha" || \
          $actual_m7b_plan_sha != "$expected_m5_m7b_plan_sha" ]]; then
        echo "M0-M7B plan preservation gate failed" >&2
        exit 1
    fi

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M6/summary.json").read_text())
if summary.get("milestone") != "M6" or summary.get("state") != "accepted":
    raise SystemExit("M6 accepted prerequisite is missing")
PY

    output="$repo_root/artifacts/M7B/ltx2-m7b-cinematic-golden-hour.mp4"
    if [[ ! -s $output || ! -s artifacts/M7B/delivery.json ]]; then
        echo "M7B delivery candidate or artifact reference is missing" >&2
        exit 1
    fi

    dotnet build Ltx.sln --configuration Release --nologo
    .venv/bin/ruff check --ignore PLR0917 \
        scripts/remote/shard-safetensors.py \
        scripts/remote/record-m7b-delivery.py \
        scripts/remote/verify-m7b.py \
        packages/ltx-core/src/ltx_core/block_streaming/disk.py \
        packages/ltx-core/src/ltx_core/loader/helpers.py \
        packages/ltx-core/src/ltx_core/quantization/fp8_cast.py \
        packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py
    bash -n scripts/remote/run-m7b-sample.sh
    python3 -m py_compile \
        scripts/remote/shard-safetensors.py \
        scripts/remote/record-m7b-delivery.py \
        scripts/remote/verify-m7b.py \
        packages/ltx-core/src/ltx_core/block_streaming/disk.py \
        packages/ltx-core/src/ltx_core/loader/helpers.py \
        packages/ltx-core/src/ltx_core/quantization/fp8_cast.py \
        packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py \
        packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py
    ffmpeg -v error -i "$output" -map 0:v:0 -map 0:a:0 -f null -
    scripts/remote/verify-m7b.py
    git diff --check

    trap - ERR
    scripts/remote/publish-heartbeat.sh M7B verified none
    echo "M7B accepted"
    exit 0
fi

if [[ $milestone == M7A ]]; then
    failure_heartbeat() {
        scripts/remote/publish-heartbeat.sh M7A blocked m7a_acceptance_gate_failed || true
    }
    trap failure_heartbeat ERR

    expected_m0_plan_sha=d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6
    expected_m1_m4_plan_sha=7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea
    expected_m5_m6_plan_sha=c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2
    expected_m7a_plan_sha=85da94394a9e37749d579bc333e79eedebe2a61f77324cacd7ecbd5d149247ba
    [[ $(sha256sum C_SHARP_PORT_PLAN.M0.md | awk '{print $1}') == "$expected_m0_plan_sha" ]]
    for plan in C_SHARP_PORT_PLAN.M1.md C_SHARP_PORT_PLAN.M2.md C_SHARP_PORT_PLAN.M3.md C_SHARP_PORT_PLAN.M4.md; do
        [[ $(sha256sum "$plan" | awk '{print $1}') == "$expected_m1_m4_plan_sha" ]]
    done
    [[ $(sha256sum C_SHARP_PORT_PLAN.M5.md | awk '{print $1}') == "$expected_m5_m6_plan_sha" ]]
    [[ $(sha256sum C_SHARP_PORT_PLAN.M7A.md | awk '{print $1}') == "$expected_m7a_plan_sha" ]]

    python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/M6/summary.json").read_text())
if summary.get("milestone") != "M6" or summary.get("state") != "accepted":
    raise SystemExit("M6 accepted prerequisite is missing")
PY

    artifact_dir="$repo_root/artifacts/M7A"
    mkdir -p "$artifact_dir"
    dotnet build Ltx.sln --configuration Release --nologo
    scripts/remote/run-m1-smoke.sh "$artifact_dir/abi-smoke.json"
    scripts/remote/run-m1-parity.sh "$artifact_dir/foundation-parity.json"
    scripts/remote/run-m2-storage.sh "$artifact_dir/storage.json"
    scripts/remote/run-m3-core.sh "$artifact_dir/core-model.json"
    scripts/remote/run-m4-cuda.sh "$artifact_dir/cuda-quantization-lora.json"
    scripts/remote/run-m5-media.sh "$artifact_dir/media-pipelines.json"
    scripts/remote/run-m6-training.sh "$artifact_dir/low-vram-training.json"
    scripts/remote/run-m7a-checkpoint.sh "$artifact_dir"
    git diff --check

    python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/M7A")
load = lambda name: json.loads((root / name).read_text())
reports = {
    "smoke": load("abi-smoke.json"),
    "foundation": load("foundation-parity.json"),
    "storage": load("storage.json"),
    "core": load("core-model.json"),
    "m4": load("cuda-quantization-lora.json"),
    "m5": load("media-pipelines.json"),
    "m6": load("low-vram-training.json"),
    "download": load("model-download.json"),
    "oracle": load("checkpoint-oracle.json"),
    "runtime": load("checkpoint-runtime.json"),
    "pipelines": load("real-pipelines.json"),
    "trainer": load("standard-trainer.json"),
    "hardware": load("hardware.json"),
}
if any(report.get("result") != "pass" for report in reports.values() if report is not reports["oracle"]):
    raise SystemExit("M7A result artifact is not passing")
expected_totals = {
    "foundation": 2,
    "storage": 16,
    "core": 34,
    "m4": 45,
    "m5": 52,
    "m6": 21,
    "runtime": 6,
    "pipelines": 12,
    "trainer": 1,
}
for name, passed in expected_totals.items():
    if reports[name].get("test_totals") != {"passed": passed, "failed": 0, "skipped": 0}:
        raise SystemExit(f"M7A {name} totals mismatch")

downloaded = {item["path"]: item for item in reports["download"].get("files", [])}
expected_files = {
    "ltx-2.3-22b-distilled-1.1.safetensors": (46149345334, "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc"),
    "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors": (327309312, "c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e"),
    "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors": (12583096, "78bffa6049bae2649a4365ec8769db88052c21348d643e8fc1ce6d483d994c5b"),
}
if downloaded.keys() != expected_files.keys():
    raise SystemExit("M7A pinned checkpoint file set mismatch")
for path, (size, checksum) in expected_files.items():
    item = downloaded[path]
    if item.get("size_bytes") != size or item.get("sha256") != checksum or item.get("state") not in {"cached_verified", "downloaded_verified"}:
        raise SystemExit(f"M7A checkpoint verification failed for {path}")

oracle = reports["oracle"]
if oracle.get("fixture_revision") != "m7a-real-checkpoint-runtime-v4" or oracle.get("tolerances") != {
    "bf16": {"rtol": 0.02, "atol": 0.005}
}:
    raise SystemExit("M7A retained checkpoint oracle mismatch")
runtime = reports["runtime"]
if runtime.get("checkpoint_model_version") != "2.3.0" or runtime.get("checkpoint_tensor_count") != 5947 or \
        runtime.get("execution") != "native_csharp_torchsharp_cuda_full_checkpoint" or \
        runtime.get("python_runtime_calls") != 0 or runtime.get("fixture_revision") != "m7a-real-checkpoint-runtime-v4":
    raise SystemExit("M7A native checkpoint runtime contract mismatch")
comparisons = runtime.get("comparisons", {})
for layer in range(48):
    for modality in ("video", "audio"):
        if comparisons.get(f"transformer_{modality}_block_{layer}", {}).get("result") != "pass":
            raise SystemExit(f"M7A transformer block parity missing for {modality} layer {layer}")
for name in ("transformer_video_velocity", "transformer_audio_velocity", "convolutional_video_decoder", "audio_vae_decoder", "vocoder_with_bandwidth_extension"):
    if comparisons.get(name, {}).get("result") != "pass":
        raise SystemExit(f"M7A checkpoint component parity missing: {name}")

surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
expected_modules = {item["module"] for item in surface["pipelines"] if item["status"] == "in_scope"}
pipelines = reports["pipelines"]
actual_modules = {item["module"] for item in pipelines.get("modes", [])}
if len(expected_modules) != 12 or actual_modules != expected_modules or \
        pipelines.get("checkpoint_model_version") != "2.3.0" or \
        pipelines.get("execution") != "native_csharp_torchsharp_cuda_full_checkpoint" or \
        pipelines.get("python_runtime_calls") != 0 or \
        any(item.get("execution") != "native_csharp_full_checkpoint" or item.get("result") != "pass" for item in pipelines.get("modes", [])):
    raise SystemExit("M7A does not execute every in-scope pipeline through the native checkpoint runtime")

trainer = reports["trainer"]
required_trainer = {
    "execution": "native_csharp_full_checkpoint_backward",
    "python_runtime_calls": 0,
    "profile": "standard_single_gpu",
    "checkpoint_model_version": "2.3.0",
    "layer_count": 48,
    "batch_size": 1,
    "mixed_precision": "bf16",
    "base_quantization": "none",
    "optimizer": "adamw",
    "optimizer_state_residence": "cuda_adamw",
    "cuda_device_count": 1,
}
if any(trainer.get(key) != value for key, value in required_trainer.items()) or \
        trainer.get("trainable_parameter_count") != 8448 or trainer.get("frozen_parameter_count", 0) < 20_000_000_000 or \
        min(trainer.get("loss", 0), trainer.get("gradient_norm", 0), trainer.get("parameter_delta_norm", 0)) <= 0:
    raise SystemExit("M7A standard-profile trainer contract mismatch")

gpu = reports["hardware"].get("gpu", {})
host = reports["hardware"].get("host", {})
if not (gpu.get("name", "").startswith("NVIDIA A100") or gpu.get("name", "").startswith("NVIDIA H100")) or \
        gpu.get("device_count") != 1 or gpu.get("memory_total_mib", 0) < 80_000 or \
        gpu.get("peak_memory_used_mib", 0) >= gpu.get("memory_total_mib", 0) or \
        host.get("minimum_memory_available_mib", 0) < 8_192:
    raise SystemExit("M7A 80 GB single-GPU/host-memory gate failed")
PY

    python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/M7A")
load = lambda name: json.loads((root / name).read_text())
smoke = load("abi-smoke.json")
foundation = load("foundation-parity.json")
storage = load("storage.json")
core = load("core-model.json")
m4 = load("cuda-quantization-lora.json")
m5 = load("media-pipelines.json")
m6 = load("low-vram-training.json")
download = load("model-download.json")
oracle = load("checkpoint-oracle.json")
runtime = load("checkpoint-runtime.json")
pipelines = load("real-pipelines.json")
trainer = load("standard-trainer.json")
hardware = load("hardware.json")
preflight = json.loads(Path("artifacts/M0/preflight.json").read_text())
surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.json"))
    if path.name != "summary.json"
}
deferred_pipelines = [item for item in surface["pipelines"] if item["status"] == "deferred"]
deferred_kernels = [item for item in surface["native_kernels"] if item["status"] == "deferred"]
summary = {
    "schema_version": 1,
    "milestone": "M7A",
    "state": "accepted",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_reference_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "plan_revision": "M7A",
    "plan_sha256": "85da94394a9e37749d579bc333e79eedebe2a61f77324cacd7ecbd5d149247ba",
    "verification_command": "scripts/remote/verify-milestone.sh M7A",
    "commands": [
        {"command": "dotnet build Ltx.sln --configuration Release --nologo", "exit_code": 0},
        *[{"command": f"scripts/remote/run-m{index}-{name}.sh", "exit_code": 0} for index, name in ((1, "smoke"), (1, "parity"), (2, "storage"), (3, "core"), (4, "cuda"), (5, "media"), (6, "training"))],
        {"command": "scripts/remote/run-m7a-checkpoint.sh", "exit_code": 0},
    ],
    "toolchain": preflight["toolchain"],
    "gpu": hardware["gpu"],
    "host": hardware["host"],
    "abi": smoke["abi"],
    "test_totals": {"passed": 192, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "cuda_quantization_lora_regression": "45_pass",
        "media_pipeline_regression": "52_pass",
        "low_vram_training_regression": "21_pass",
        "real_checkpoint_components": "5_pass",
        "real_checkpoint_single_gpu_pipelines": "12_pass",
        "standard_profile_trainer": "1_pass",
    },
    "checkpoint": {
        "model_version": runtime["checkpoint_model_version"],
        "tensor_count": runtime["checkpoint_tensor_count"],
        "files": download["files"],
        "oracle_revision": runtime["fixture_revision"],
        "execution": runtime["execution"],
        "python_runtime_calls": runtime["python_runtime_calls"],
    },
    "real_pipelines": pipelines["modes"],
    "standard_trainer": trainer,
    "numerical_tolerances": runtime["numerical_tolerances"],
    "deferred": {
        "pipelines": deferred_pipelines,
        "native_kernels": deferred_kernels,
        "multi_gpu": "out_of_scope",
        "ddp_fsdp_nccl_cuda_ipc_all_to_all": "out_of_scope",
        "b200_specific": "out_of_scope",
        "full_fine_tuning": "out_of_scope",
        "hopper_specific_fp8_on_a100": "out_of_scope",
    },
    "artifact_sha256": checksums,
    "secrets_recorded": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    trap - ERR
    echo "M7A accepted"
    exit 0
fi

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
    (m5, {"passed": 52, "failed": 0, "skipped": 0}),
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
    "test_totals": {"passed": 174, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "cuda_quantization_lora_regression": "45_pass",
        "media_pipeline_regression": "52_pass",
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
if m5.get("test_totals") != {"passed": 52, "failed": 0, "skipped": 0}:
    raise SystemExit("M5 media/pipeline totals mismatch")
if m5.get("suite_totals") != {
    "media": 10,
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
    "test_totals": {"passed": 153, "failed": 0, "skipped": 0},
    "tests": {
        **smoke["tests"],
        "foundation_parity": "2_pass",
        "storage_regression": "16_pass",
        "core_model_regression": "34_pass",
        "cuda_quantization_lora_regression": "45_pass",
        "media": "10_pass",
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
