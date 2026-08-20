#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
report=${1:-$repo_root/artifacts/M5/media-pipelines.json}
fixture="$repo_root/tests/fixtures/M5"
cd "$repo_root"

fixture_temp=$(mktemp -d /tmp/ltx-m5-fixture.XXXXXX)
cli_temp=$(mktemp -d /tmp/ltx-m5-cli.XXXXXX)
cleanup() {
    if [[ $fixture_temp == /tmp/ltx-m5-fixture.* && -d $fixture_temp ]]; then
        rm -rf -- "$fixture_temp"
    fi
    if [[ $cli_temp == /tmp/ltx-m5-cli.* && -d $cli_temp ]]; then
        rm -rf -- "$cli_temp"
    fi
}
trap cleanup EXIT

"$repo_root/.venv/bin/python" scripts/parity/generate-m5-fixtures.py "$fixture_temp"
tracked_files=$(find "$fixture" -type f -printf '%P\n' | sort)
generated_files=$(find "$fixture_temp" -type f -printf '%P\n' | sort)
if [[ $tracked_files != "$generated_files" ]]; then
    echo "tracked M5 fixture file set does not match the pinned oracle" >&2
    exit 1
fi
while IFS= read -r name; do
    if ! cmp -s "$fixture/$name" "$fixture_temp/$name"; then
        echo "tracked M5 fixture '$name' does not match the pinned oracle" >&2
        exit 1
    fi
done <<<"$tracked_files"

"$repo_root/.venv/bin/python" - <<'PY'
import json
from pathlib import Path

surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
manifest = json.loads(Path("tests/fixtures/M5/manifest.json").read_text())
in_scope = [item for item in surface["pipelines"] if item["status"] == "in_scope"]
deferred = [item for item in surface["pipelines"] if item["status"] == "deferred"]
fixture_modules = [item["module"] for item in manifest["pipeline_modes"]]
inventory_modules = [item["module"] for item in in_scope]
if fixture_modules != inventory_modules or len(in_scope) != 12:
    raise SystemExit("M5 fixture does not cover the exact M0 in-scope pipeline inventory")
if len(deferred) != 3 or any("_mgpu" not in item["module"] for item in deferred):
    raise SystemExit("M5 deferred pipeline inventory differs from the accepted M0 inventory")
entrypoints = {
    item["name"]: item for item in surface["cli_entrypoints"] if item["name"].startswith("ltx_pipelines.")
}
if any(module not in entrypoints or entrypoints[module]["status"] != "in_scope" for module in fixture_modules):
    raise SystemExit("M5 pipeline CLI coverage differs from the M0 CLI inventory")
PY

scripts/remote/build-m5-media.sh >/dev/null
export LD_LIBRARY_PATH="$repo_root/build/M5/ltx-media${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

media_output=$(dotnet run \
    --project tests/Ltx.MediaPipelineTests/Ltx.MediaPipelineTests.csproj \
    --configuration Release \
    --no-build \
    -- "$fixture")
if [[ $media_output != *"M5 media/pipeline fixtures: 21 passed, 0 failed; media=9, pipelines=12"* ]]; then
    echo "M5 C# runner did not report the complete twenty-one-check pass" >&2
    exit 1
fi

cli_report="$cli_temp/cli.json"
cli_output=$("$repo_root/.venv/bin/python" scripts/parity/run-m5-cli-fixtures.py \
    src/Ltx.Cli/bin/Release/net10.0/Ltx.Cli.dll "$fixture" "$cli_temp/output" "$cli_report")
if [[ $cli_output != *"M5 CLI fixtures: 30 passed, 0 failed; modes=12, help=12, errors=6"* ]]; then
    echo "M5 CLI runner did not report the complete thirty-check pass" >&2
    exit 1
fi

mkdir -p "$(dirname "$report")"
REPORT="$report" FIXTURE="$fixture" CLI_REPORT="$cli_report" MEDIA_OUTPUT="$media_output" CLI_OUTPUT="$cli_output" \
"$repo_root/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

fixture = Path(os.environ["FIXTURE"])
manifest = json.loads((fixture / "manifest.json").read_text())
surface = json.loads(Path("artifacts/M0/source-surface.json").read_text())
cli = json.loads(Path(os.environ["CLI_REPORT"]).read_text())
checksums = {
    str(path.relative_to(fixture)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(fixture.rglob("*"))
    if path.is_file()
}
digest = hashlib.sha256()
for name, checksum in sorted(checksums.items()):
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(checksum.encode())
    digest.update(b"\0")

in_scope = [item for item in surface["pipelines"] if item["status"] == "in_scope"]
deferred = [item for item in surface["pipelines"] if item["status"] == "deferred"]
mode_results = {item["module"]: "seeded_fixture_and_cli_pass" for item in in_scope}
native_library = Path("build/M5/ltx-media/libltx_oiio.so")
report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "result": "pass",
    "fixture_revision": manifest["fixture_revision"],
    "fixture_set_sha256": digest.hexdigest(),
    "fixture_sha256": checksums,
    "oracle": manifest["oracle"],
    "dependencies": {
        "dotnet_sdk": subprocess.check_output(["dotnet", "--version"], text=True).strip(),
        "ffmpeg": manifest["oracle"]["ffmpeg"],
        "openimageio": manifest["oracle"]["openimageio"],
        "native_oiio_bridge_sha256": hashlib.sha256(native_library.read_bytes()).hexdigest(),
    },
    "test_totals": {"passed": 51, "failed": 0, "skipped": 0},
    "suite_totals": {
        "media": 9,
        "pipelines": 12,
        "cli_modes": 12,
        "cli_help": 12,
        "cli_errors": 6,
    },
    "pipeline_inventory": {
        "total": len(surface["pipelines"]),
        "in_scope": len(in_scope),
        "tested": len(mode_results),
        "correctness": mode_results,
        "deferred": deferred,
    },
    "cli_behavior": cli,
    "media": {
        "video": "ffmpeg_mp4_decode_encode_pass",
        "audio": "pcm_wave_and_container_audio_pass",
        "images": "png_decode_pass",
        "hdr": "rec2020_hlg_tags_and_tonemap_pass",
        "exr": "openimageio_native_read_write_sequence_pass",
        "hdr_ic_lora_output": "nine_exr_frames_and_prores_mov_pass",
    },
    "numerical_tolerances": {
        "fp32": {**manifest["tolerances"]["fp32"], "result": "pass"},
    },
    "output": {
        "managed": os.environ["MEDIA_OUTPUT"],
        "cli": os.environ["CLI_OUTPUT"],
    },
    "secrets_recorded": False,
}
Path(os.environ["REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n%s\n' "$media_output" "$cli_output"
