# Continuing the remote C# LTX port

This runbook resumes work from accepted fork branch `codex/ltx-csharp` at `3bafc887cee891ba7315a87f422f45a2107abe2b`. It assumes no existing Vast instance, no persistent volume, and no surviving deploy key from the completed run.

## 1. Choose the next work item and its acceptance oracle

Begin from the deferred-work list in `C_SHARP_PORT_COMPLETION_REPORT.md`. For any workflow beyond component fixtures, define the acceptance command and a paired C#-then-Python oracle before renting a GPU:

- fix prompt, seed, checkpoint revisions, frame count, FPS, resolution, sampler, and offload mode;
- run C# first; a failed C# run is a defect to diagnose, never a reason to substitute Python output;
- run the Python reference with the same inputs and record any intentional runtime-mode difference;
- retain both MP4s or equivalent outputs, SHA-256 values, `ffprobe`/decode results, peak memory, elapsed time, and comparison metrics;
- update `PORT_STATUS.md` only with a SHA that has already been pushed to the fork.

Use `scripts/remote/verify-milestone.sh <milestone>` as the remote acceptance entry point. Add the new oracle to it or to a successor milestone verifier; do not rely on a manually observed successful console run.

For any memory/performance work, capture a stage-by-stage C# versus Python allocation profile: checkpoint loading, resident weight storage, tensor lifetime/disposal, duplicate buffers, CPU/GPU copies, activation peaks, allocator behavior, streaming/offload/cache scheduling, precision, attention, and VAE tiling. Treat C# object layout or GC as a hypothesis to test, not the presumed cause.

## 2. GitHub preparation

1. Clone the fork remotely, not on the local coordinator: `https://github.com/miloszkukla/LTX-2.git`.
2. Start from `codex/ltx-csharp`; create a `codex/<bounded-work-item>` branch only when the work needs isolation.
3. Generate a fresh SSH deploy key on the remote worker. Register only its public half as a write deploy key on `miloszkukla/LTX-2`; keep the private half on that worker and never copy it into chat, source, images, or the local machine.
4. Prove one small remote push before expensive model downloads. Record the key ID in redacted status, then revoke that specific key after the worker is clean and all commits are pushed.
5. Do not assume GitHub Packages/Container Registry is anonymously usable. The existing Actions base-image workflow succeeded, but anonymous GHCR pull was not proven. A free GitHub account is not itself a blocker; first test an anonymous pull by the exact image digest from a fresh worker. If it fails, use the checked-in NVIDIA-base bootstrap path below.

## 3. Hugging Face preparation

Use a fine-grained read-only Hugging Face token and accept access conditions in the browser for every required repository before starting the worker. The completed run needed access beyond the main LTX checkpoint, including Pixel Upscaler, LTX 2.3 compatibility artifacts, HDR, DubIt, and Gemma-related artifacts. Recheck the exact current artifact manifest before a new workflow.

Configure the token as the Vast account `HF_TOKEN` environment variable. Do not put it in Docker `ARG`/`ENV`, shell profiles, tmux configuration, source files, Git history, logs, or a command line that may be recorded.

On SSH-runtype workers, account variables can be present only in PID 1’s environment. For one command that needs Hugging Face access, load it transiently without printing it:

```bash
export HF_TOKEN="$(tr '\0' '\n' </proc/1/environ | sed -n 's/^HF_TOKEN=//p' | head -n1)"
test -n "$HF_TOKEN" || { echo 'HF_TOKEN unavailable to this worker'; exit 1; }
```

Run read-only authenticated access probes for every required artifact before downloading model data. Record model revisions, expected sizes, and checksum sources in redacted evidence.

## 4. Hardware and Vast.ai gates

Use a single GPU. Multi-GPU remains a future feature and must not be silently enabled as a workaround.

| Workload | Minimum practical target | Gate |
| --- | --- | --- |
| Small fixtures and low-VRAM development | RTX 6000 Ada 48 GB preferred | Cold-start the exact intended workload; do not infer capacity from model name or a Python-only run. |
| Current C# standard-profile two-stage video and C#/Python parity | A100 80 GB preferred; H100 80 GB when Hopper behavior is specifically needed | Treat 80 GB as the baseline until profiling lowers the current C# peak. The accepted C# run peaked at 81,147 MiB with disk-weight streaming. |
| System/disk | 64 GB+ host RAM, 200 GB real disk minimum | Verify the actual post-launch disk and available RAM; recreate if model/cache manifest plus 20% headroom exceeds it. |

Current marketplace prices and names are dynamic. Search broadly by VRAM and reliability rather than relying on an old offer ID or a named-GPU filter that might omit a listing:

```powershell
vastai search offers 'num_gpus=1 gpu_ram>=48 reliability>=0.95 rentable=true direct_port_count>=1' -o dph_total --raw
```

For a standard-profile run, require `gpu_ram>=80`. Choose the lowest-cost compatible offer only after checking its actual GPU, host RAM, disk, and driver. The previous run rejected a driver below 580 and a worker whose actual disk was much smaller than requested.

Create with the documented safety flags and an explicit disk request:

```powershell
vastai create instance <offer-id> --image nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 --disk 200 --ssh --direct --cancel-unavail
```

`--direct` is a Vast provisioning flag in this workflow; the remote Codex work itself does not require an interactive direct-SSH session. Use the connection endpoint returned by `vastai ssh-url --raw` for bootstrap/control, and inspect `vastai logs <instance-id>` before attempting recovery when a launch or SSH connection fails.

Before downloading checkpoints, verify on the worker:

```bash
nvidia-smi
nvcc --version
df -h /
free -h
```

Require NVIDIA driver `>=580`, CUDA 13.x compatibility, compiled SASS for the reported GPU architecture, and a CUDA/TorchSharp/native smoke test. Never trust requested disk or advertised RAM without measuring the running machine.

## 5. Docker and remote bootstrap

Prefer the source-controlled assets:

- `Dockerfile.remote`
- `.dockerignore`
- `scripts/remote/bootstrap-host.sh`

The source-free base image must contain the CUDA development toolchain, .NET 10 SDK, CMake/Ninja, Git, Python/uv, FFmpeg, OpenCV prerequisites, and OpenImageIO prerequisites, but never repository source, model weights, SSH keys, Codex credentials, or `HF_TOKEN`.

Workflow:

1. If an anonymous pull by the exact GHCR digest succeeds from a fresh worker, use that digest.
2. Otherwise start from the NVIDIA CUDA image shown above and execute `scripts/remote/bootstrap-host.sh` from the checked-out repository.
3. Record resolved image digest, installed package versions, `ldconfig`/P-Invoke smoke-test result, driver, CUDA, GPU compute capability, disk, and RAM in the next redacted milestone evidence.

Do not bake Hugging Face credentials into an image. Keep models outside image layers and measure their real cache/disk use.

## 6. Remote Codex execution and instrumentation

Run Codex on the worker in a persistent session such as tmux. Authenticate interactively on the remote worker if required; do not export credentials to the local coordinator. For a high-complexity implementation session, the completed run used GPT-5.6 Sol with xhigh reasoning, for example:

```bash
codex exec -m gpt-5.6-sol -c 'model_reasoning_effort="xhigh"' --sandbox danger-full-access
```

Keep the remote session observable through tmux/Codex logs. The local coordinator should use an SSH tail as the normal live-progress view, for example `tail -n 100 -F <remote-log>`, rather than force-pushing periodic heartbeat commits. Check GitHub when a milestone is accepted, a commit is pushed, an artifact is delivered, or SSH recovery is required.

Retain compact, push-backed evidence at those boundaries:

- instance ID, milestone, phase, UTC timestamp, and pushed commit SHA;
- build/test exit code and totals;
- GPU memory/temperature samples;
- model revisions, artifact checksums, output checksums, and failure reason without credentials;
- clean-worktree confirmation before a handoff or teardown.

If a handoff fails because the plan input is missing, correct the control-plane path, verify its SHA without reading unrelated source, and relaunch. If launch or handoff fails twice, stop and diagnose instead of repeatedly renting or restarting machines.

## 7. Closeout

After acceptance:

1. Run the milestone verifier and preserve its redacted summary.
2. Commit and push source/tests from the remote worker.
3. Verify the pushed SHA exists on the fork and download any user-review artifact to the local machine with its SHA-256 and playability check.
4. Keep the worker alive until the user has reviewed any requested video artifacts.
5. On explicit teardown approval: clean the worktree, log out remote Codex, revoke the recorded GitHub deploy key, then run `vastai destroy instance <instance-id> -y`.
6. Confirm the instance no longer appears in `vastai show instance <instance-id> --raw`, and report any remaining future work rather than claiming multi-GPU or untested workflows are finished.
