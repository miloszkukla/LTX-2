# C# port completion report

Date: 2026-08-20
Accepted source branch: `codex/ltx-csharp`
Accepted head: `3bafc887cee891ba7315a87f422f45a2107abe2b`

## Outcome

The approved porting plan was completed through M8, and this report is the planned M9 coordinator deliverable. The accepted integration contains the source port, remote build/bootstrap material, milestone verification, real-checkpoint validation, and paired C#/Python video evidence. The user reviewed the paired C#/Python beach-volleyball outputs and approved them. Both Vast instances, their remote Codex sessions, and their ephemeral GitHub deploy keys were subsequently removed.

This report is an audit of the accepted remote evidence, not a claim that every future LTX workflow or multi-GPU configuration is complete.

## Milestone audit

| Milestone | Result | Evidence |
| --- | --- | --- |
| M0: remote bootstrap | Completed | Accepted commit `28ad7ad1e6e1`; source-free base workflow succeeded; source-surface inventory records 15 pipelines and 29 native kernels. |
| M1: C# foundation and ABI | Completed | Accepted commit `e9dcbdf69c65`. |
| M2: storage and model loading | Completed | Accepted commit `f53c216a4a95`. |
| M3: core model execution | Completed | Accepted commit `ebbfd2b38a8e`. |
| M4: CUDA, quantization, and LoRA | Completed | Accepted commit `f23816df8527`. |
| M5: media and pipelines | Completed | Accepted commit `829c61c96166`. |
| M6: low-VRAM training | Completed | Accepted commit `87ddf3078caf`. |
| M7A: standard-profile validation | Completed | Accepted commit `523d81b92c79`; real checkpoint pipelines and standard trainer passed. |
| M7B: 32 GB delivery sample | Completed | Accepted commit `9c705da5a58f`; a Python two-stage sample was delivered and verified locally. |
| M7C: paired 80 GB C#/Python video | Completed | Accepted commit `51294b730a21`; C# ran first, then Python, under matching generation settings. |
| M8: integration, delivery, and teardown | Completed | Accepted merge/integration commit `3bafc887cee8`; `verify-milestone.sh M8` reports 267 passed, 0 failed, 0 skipped. Local video delivery was checksum-verified and user-reviewed before teardown. |
| M9: future-work report | Completed by this document | See “Remaining work” and the continuation runbook. |

The accepted status ledger and immutable handoff-plan revisions are in `PORT_STATUS.md` on the accepted branch; it was kept current through M8.

## End-to-end evidence

The final M8 verifier reran the merged acceptance surface: 192 M7A checks, 4 merged M7B source/evidence checks, 66 M7C paired-video checks, and 5 M8 evidence checks. Total: 267 passing, no failures or skips.

For the user-selected M7C prompt, both runs used seed `20260821`, 241 frames, 24 fps, 1536x1024, and a ten-second two-stage configuration. Both MP4s passed stream decode validation and contain 241 H.264 frames plus AAC audio.

| Run | Elapsed | Peak GPU memory | SHA-256 | Notes |
| --- | ---: | ---: | --- | --- |
| C# | 424.476 s | 81,147 MiB | `9bd4d680…785af` | Required disk-weight streaming after resident weights exhausted memory. |
| Python reference | 123.905 s | 48,947 MiB | `16a15b…32fd` | Used resident weights. |

The observed C#/Python wall-clock ratio was 3.43x. It is not a clean language-runtime benchmark: the C# run used disk streaming while the Python reference was resident, and the C# path included the repaired production execution route. It is useful as a baseline, not as a performance verdict.

The comparison artifacts recorded SSIM 0.869215, PSNR 23.818049 dB, and audio APSNR 171.886 dB. More importantly, the user visually approved the two outputs as indistinguishable for the tested prompt.

Local deliverables retained from the run:

- `artifacts/M7B/ltx2-m7b-cinematic-golden-hour.mp4`
- `artifacts/M7C/m7c-csharp-beach-volleyball.mp4`
- `artifacts/M7C/m7c-python-beach-volleyball.mp4`
- `artifacts/M7C/m7c-csharp-vs-python-side-by-side.mp4`

## How the plan matched reality

The plan’s remote-first control model worked: source edits, builds, tests, commits, and pushes happened on remote workers; the local side acted as the control plane and retained only redacted status/evidence plus downloaded artifacts. Milestone handoffs, isolated M7A/M7B branches, and an explicit user-review gate avoided losing accepted work or tearing down the review worker prematurely.

There were several material corrections during execution:

1. The GitHub Actions base-image build succeeded, but anonymous GHCR pullability was not established. The plan’s fallback was therefore exercised: start from the NVIDIA CUDA image and run checked-in `scripts/remote/bootstrap-host.sh`. A future worker must test an anonymous pull by digest before relying on GHCR.
2. The initial 32 GB M7B result was a useful delivery/infrastructure validation, but it was Python-only and therefore did not establish C# inference parity. The plan was correctly expanded with M7C: a C#-first, same-prompt, same-seed Python reference comparison.
3. Component and small-fixture acceptance did not catch all full-pipeline defects. The real C# run exposed RoPE-axis ordering, native linear/add parity, BF16/Euler scheduler semantics, learned x2 upscaling, tiled VAE decoding, and explicit 241-frame muxing issues. These were repaired before M7C acceptance.
4. Requested disk and actual disk must be checked after provisioning. Earlier 80 GB candidates were discarded for an insufficient driver and for receiving far less disk than requested; the final worker passed a driver `>= 580` and real-disk gate.
5. Hugging Face access must be probed for every auxiliary dependency, not just the primary checkpoint. The run required gates for the main LTX model, Pixel Upscaler, LTX 2.3 compatibility artifacts, HDR, DubIt, and Gemma-related artifacts.
6. Five-minute redacted GitHub heartbeat commits were more operational overhead than value once SSH control access existed. Future runs should tail the remote tmux/Codex logs for live progress, and use GitHub only for pushed milestone commits, final evidence, and recovery checks.

## Hardware conclusion

32 GB is not an adequate default target for the current C# standard-profile two-stage video path. It was sufficient only for the M7B Python sample with low-VRAM/offload settings.

An RTX 6000 Ada with 48 GB is a sensible minimum development target above 32 GB, but it is not proven sufficient for a resident ten-second 1536x1024 C# two-stage run. The accepted C# run peaked at 81,147 MiB even with disk-weight streaming. Until memory work materially reduces that peak, use a single 80 GB A100 or H100 as the reproducible standard-profile and parity-validation baseline; use a 48 GB RTX 6000 Ada only after a cold-start, exact-workload gate succeeds.

## Remaining work

Confirmed deferred or incomplete areas:

- Multi-GPU inference and training: NCCL/process groups, rank-aware execution, all-to-all/CUDA IPC, DDP/FSDP, sharded checkpoints, and multi-GPU acceptance tests remain out of scope.
- End-to-end oracle breadth: add fixed prompt/seed C#-then-Python paired tests for every in-scope workflow and relevant VRAM/configuration tier, especially production scheduler, attention, offload, tiled decoding, media muxing, audio, HDR/EXR, retake, interpolation, IC-LoRA, and training paths. Component fixtures alone are insufficient.
- Performance: profile C# and Python GPU allocations stage-by-stage: checkpoint loading, resident weights, tensor lifetime/disposal, duplicate buffers, CPU/GPU copies, activation peaks, allocator behavior, streaming/offload/cache scheduling, precision, attention, native kernels, compilation, model-cache reuse, and VAE tiling. Optimize confirmed contributors rather than assuming ordinary C# GC or data structures are the root cause. Re-run a fair C#/Python comparison only after matching memory modes and cache conditions.
- Portability/deployment: prove or replace the anonymous GHCR distribution path; keep the NVIDIA-base bootstrap fallback tested. Validate the supported CUDA/driver matrix on newly rented hosts.
- Checkpoint coverage: add a manifest-driven model-access gate and explicit artifact revision/checksum records to every future full-pipeline run.

## Cleanup confirmation

The 32 GB worker `48162892` and 80 GB validation worker `48178709` were destroyed after the user’s review approval. The remote Codex logout/clean-worktree check passed on the final 80 GB worker before destruction. The associated remote-only GitHub deploy keys were revoked; the repository has no remaining deployment keys from this run.
