# C# port status

## Plan revision history

- M0 bootstrap plan: `C_SHARP_PORT_PLAN.M0.md`, SHA-256 `d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6`. This is the byte-for-byte plan accepted at M0 and remains immutable historical evidence.
- Accepted M1 plan: `C_SHARP_PORT_PLAN.M1.md`, revision M1, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `e9dcbdf69c658ad026bdec6ee5cd4642847215a8`.
- Accepted M2 plan: `C_SHARP_PORT_PLAN.M2.md`, revision M2, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `f53c216a4a9598d1bc626923744095aac7311a46`.
- Accepted M3 plan: `C_SHARP_PORT_PLAN.M3.md`, revision M3, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `ebbfd2b38a8eb5fd346ffe9b3e3833bab05277d4`.
- Accepted M4 plan: `C_SHARP_PORT_PLAN.M4.md`, revision M4, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `f23816de4e48905904d72bce02924a88dad9705a`; the coordinator's M4 revision was content-identical to the accepted M1, M2, and M3 revisions.
- Accepted M5 plan: `C_SHARP_PORT_PLAN.M5.md`, revision M5, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`. This preserves the byte-for-byte plan accepted with checkpoint `829c61c8d3b9733ae9b813bdf7b0dc892e978a7e` and includes the approved M7A/M7B/M8 delivery and teardown workflow.
- Accepted M6 plan: revision M6, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`. Its byte-for-byte contents remain preserved by accepted checkpoint `87ddf3078caf1d41290ee514b6c3c61e06448160`; the coordinator's M6 revision was content-identical to the accepted M5 revision.
- Accepted M7B plan: `C_SHARP_PORT_PLAN.M7B.md`, revision M7B, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`. It preserves the content-identical coordinator handoff accepted by checkpoint `9c705da5a58f37c4d1e75b19a0fa1a9e6a216597`.
- Accepted M7A plan: `C_SHARP_PORT_PLAN.M7A.md`, revision M7A, SHA-256 `85da94394a9e37749d579bc333e79eedebe2a61f77324cacd7ecbd5d149247ba`. It preserves the byte-matched coordinator handoff accepted on the isolated M7A branch.
- Authorized M7C plan: `/root/C_SHARP_PORT_PLAN.M7C.md`, revision M7C, SHA-256 `6c7fbd1012661adba0dc7a9b89a8cbed7c1a1014fa9475b93531edad58578982`. M7C was performed on accepted M7A source checkpoint `523d81b92c796048f2bbcc299c5c4f21e3fd5b5e` and preserves the required C#-first paired-video order.
- Current M8 plan: `C_SHARP_PORT_PLAN.md`, revision M8, SHA-256 `dbe05843ca8f2da269b0703469e222edfb72b55a9bbc824af0fa6bd882d8a157`. It byte-matches `/root/C_SHARP_PORT_PLAN.M8.md`, retains the user-review teardown gate, and requires non-destructive integration and verification on this worker.

## M8 — accepted non-destructive delivery checkpoint

- Integration: accepted M7B/source parent `9c705da5a58f37c4d1e75b19a0fa1a9e6a216597` and accepted M7A+M7C/source parent `51294b730a211b643118a4013f4b599d57a78de6` were combined in one two-parent merge on `codex/ltx-csharp`. No implementation file had a textual conflict; the four expected status/heartbeat/verifier conflicts were resolved by preserving all M7A, M7B, and M7C paths and the M7A worker's isolated deploy identity/status branch.
- Verification: `scripts/remote/verify-milestone.sh M8` (exit 0), the sole M8 gate. It reran the complete M7A standard-profile suite, checked the merged M7B Python sources and accepted redacted evidence, reran the complete M7C paired-video regression, checked the merge parents and plan hashes, and emitted `artifacts/M8/summary.json`.
- Results: 267 checks passed, 0 failed, 0 skipped: 192 M7A checks (173 prerequisite regressions, 6 production-checkpoint runtime checks, 12 real pipeline routes, and 1 standard trainer), 4 merged-M7B source/evidence checks, 66 M7C checks, and 5 M8 integration/evidence checks. Release builds completed with 0 warnings and 0 errors; all numerical tolerances remain unchanged.
- Merge repair provenance: the first integrated gate exposed only stale M7A accounting after M7C added the sixth runtime regression; the executable runtime had already passed 6/6. The expectation and M7A total were corrected from 5/191 to 6/192. The next attempt found the M7B worker's `ruff` executable absent from this host, so the declared `ruff>=0.14.3` prerequisite was installed locally as `ruff 0.16.3`; the final unchanged gate then passed.
- Verified source: pre-evidence merge checkpoint `7ba03274afa5f5ba78e26a70e3b6d75fa9bf4b13`, source-content manifest SHA-256 `d60a9b69aca996a5e073499ee5e699f2ab3b09fc2d2f816500c0986311fd8a70`. The final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- M7C delivery confirmation: the coordinator explicitly confirmed that both locally copied MP4s checksum-match the accepted evidence. Native C# SHA-256 is `9bd4d680246737dcac62226b9c5c1984cb9f1d093bfec5bd0704fc952a1785af`; Python reference SHA-256 is `16a15b7e68d94087da1ed1c3217e22dfacdaa5ceedda21ab83c5ff6b24ac32fd`. Both retained on-host files were fully decoded again during M8.
- Completion evidence: `artifacts/M8/summary.json`, SHA-256 `6f3b1f10b56fae4b63e276114e1845b20862a4e0ad70940f942ad8b7d91bc56e`, plus refreshed M7A and M7C redacted reports. The same M8 command performs the final read-only source-manifest, clean-worktree, exact-remote-tip, and heartbeat audit after the atomic push.
- Teardown gate: closed pending explicit user review authorization. No Codex logout, deploy-key revocation, instance destruction, artifact deletion, or other destructive action was performed. Known failures: none after the recorded repairs. The 80 GB worker remains retained and will be left idle with the recurring accepted M8 heartbeat.

## M7B — accepted cinematic delivery checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M6 checkpoint `87ddf3078caf1d41290ee514b6c3c61e06448160`; no M7B implementation or model download began before pushed documentation/status checkpoint `5f403e3206b2c148be5f717038a89cb8a2a718d5`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M7B.md`, revision M7B, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`; retained as `C_SHARP_PORT_PLAN.M7B.md`. All accepted M0–M6 plan and milestone evidence is preserved in the tracked historical copies, accepted commits, and sections below.
- Verification: `scripts/remote/verify-milestone.sh M7B` (exit 0). The Release solution build completed with 0 warnings and 0 errors; Python lint/compile, shell syntax, all six pinned model payload hashes, shard provenance, fallback policy, GPU-memory evidence, MP4 hash/probe, full video/audio decode, and local-delivery reference passed 14 checks with 0 failed and 0 skipped.
- Prerequisite recovery: the earlier redacted 403 checkpoint `43130bef3bdacec151040dbbd18891746d83a14d` is preserved. Subsequent authenticated byte-range access succeeded, and all required files were downloaded from exactly `Lightricks/LTX-2.5@6c7e5e573ac1667efc83407806fe9b0b93730e60`; their six accepted M0 SHA-256 values passed again during final verification.
- Diagnosed first attempt: the published 42,018,190,584-byte transformer could not be whole-file mmaped inside the worker's 31,403,802,624-byte host-memory cgroup, before inference or material VRAM allocation. The payload was copied byte-for-byte into five independently mmap-able transformer shards, the 26,263,858,182-byte text encoder into four shards, and the disk streamer was updated to keep at most one shard mmap active. Source payload hashes, tensor counts, shard hashes, and manifests are recorded in the delivery evidence.
- Preferred-path result: `TI2VidTwoStagesPipeline` completed at the original 241-frame, 24 fps, 1536×1024, 30-step profile with the full 22B development transformer, distilled LoRA strength 1.0, FP8 cast, batch size 1, disk offload, and chunked-eager DiffVAE. Runtime was 4,943 seconds and peak sampled GPU memory was 26,090 MiB of 32,607 MiB on the single RTX 5090. No duration/resolution mitigation and no standalone distilled fallback were used.
- Cinematic output: H.264 `yuv420p` video plus stereo 48 kHz AAC audio, 241 frames at 24 fps, duration 10.041667 seconds, 9,445,404 bytes. The full audio/video decode and visual beginning/middle/end continuity inspection passed.
- Local delivery: `/workspace/LTX-2/artifacts/M7B/ltx2-m7b-cinematic-golden-hour.mp4`, SHA-256 `3902bf2f534ca7409766e24c620921ecc22950bdc3b70fdc5c7a2e2a7c210ee9`. It is ready for the coordinator to copy only after this accepted checkpoint is pushed; the coordinator must verify the matching hash after copying.
- Redacted evidence: `artifacts/M7B/prerequisite-failure.json`, both two-stage attempt records, `artifacts/M7B/delivery.json`, and `artifacts/M7B/summary.json`. The accepted summary records the prompt, seed `20260820`, exact commands/settings, dependencies, model revision/hashes, diagnosed failure and mitigation, media probe, GPU peak, output hash, and local-delivery condition without secrets.
- Starting/source checkpoint: `43130bef3bdacec151040dbbd18891746d83a14d`; the final accepted checkpoint SHA is recorded by the redacted status branch immediately after its atomic push because it cannot be embedded in its own commit.
- Known failures: none. M7B is accepted and its source checkpoint is an M8 merge parent.

## M7C — accepted paired-video checkpoint

- Verification: `scripts/remote/verify-milestone.sh M7C` (exit 0), the sole M7C gate. It reverified the plan and model hashes, rebuilt Release and the maintained native bridge, ran 66 checks with 0 failures and 0 skips, fully decoded both final MP4s, and validated every redacted evidence report.
- Exact prompt: “Two adult women with long blonde hair play a lively beach-volleyball rally on a sunny beach. They wear sporty bikinis appropriate for beach volleyball and speak casually to each other between plays, smiling and calling the ball. Natural, realistic athletic motion; ocean waves, warm sand, gentle sea breeze, distant beach ambience, synchronized dialogue and sound. Cinematic tracking camera, no cuts.” Prompt SHA-256 is `fe9c3956eff10418967a912607b3c29ae77b9920a7013f62e692fa06742d855d`.
- Execution order and settings: the native C# two-stage pipeline completed and produced a validated container before the Python reference pipeline started. Both used seed `20260821`, 241 frames, 24 fps, 1536×1024, 251 audio tokens, stage-1 sigmas `[1,.99375,.9875,.98125,.975,.909375,.725,.421875,0]`, and stage-2 sigmas `[.909375,.725,.421875,0]`.
- Native C# output: `build/M7C/videos/m7c-csharp-beach-volleyball.mp4`, SHA-256 `9bd4d680246737dcac62226b9c5c1984cb9f1d093bfec5bd0704fc952a1785af`, 18,279,731 bytes, 424.475850 seconds generation time. FFprobe reports H.264 1536×1024, 241 frames at 24 fps, 10.041667 seconds, plus stereo 48 kHz AAC (10.010 seconds); a complete decode emitted no errors.
- Python reference output: `build/M7C/videos/m7c-python-beach-volleyball.mp4`, SHA-256 `16a15b7e68d94087da1ed1c3217e22dfacdaa5ceedda21ab83c5ff6b24ac32fd`, 8,257,254 bytes, 123.904568 seconds generation time. FFprobe reports the same H.264 geometry/frame contract and stereo 48 kHz AAC duration; a complete decode emitted no errors.
- Memory/offload: the worker is one NVIDIA A100-SXM4-80GB (81,920 MiB, compute capability 8.0). The preferred resident C# attempt reached CUDA OOM and was retained for diagnosis, so the successful C# run used required safetensor disk-weight streaming and peaked at 81,147 MiB. Python remained fully resident with no offload and peaked at 48,947 MiB. Shared Gemma prompt conditioning also ran resident, peaked at 51,065,139,712 allocated bytes, and wrote the exact common contexts.
- Pinned models: `Lightricks/LTX-2.3@6b5a83e3045eaf8e46cfa0acce512412aa2b9cce`, monolithic checkpoint SHA-256 `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`; learned x2 spatial upsampler SHA-256 `5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed`; and `google/gemma-3-12b-it-qat-q4_0-unquantized@68f7ee4fbd59087436ada77ed2d62f373fdd4482`.
- Port repair: the first playable C# output had collapsed beige content. Layer-level production diagnostics localized the defect to RoPE frequency-axis ordering: C# transposed axes that were already in reference layout. Removing that transpose raised first-step final video/audio velocity cosine to `0.99997896`/`0.99998534`. Supporting repairs added exact native `at::linear`/`at::add`, Python SDPA priority, X0 BF16/Euler semantics, the learned spatial upsampler, bounded exact decoder tiling, and explicit 241-frame muxing. Python was used only as the required reference/oracle and never substituted for the C# pipeline.
- Comparison: the five-time contact sheet was visually reviewed as a coherent matching sunny-beach scene with the same two adult blonde athletes and continuous composition. Full-video C# versus Python metrics are SSIM `0.869215` and PSNR `23.818049` dB; both audio channels have APSNR `171.886` dB. The playable side-by-side artifact is `build/M7C/comparison-final/m7c-csharp-vs-python-side-by-side.mp4`, SHA-256 `ab6258f332e5ed861603a58f3f0245c5b571088565ea41588db08fa69314678f`; the contact sheet SHA-256 is `aa78f94e16c87743188d91bcb705cbc0d7e87ad4023399f7d5eab6eb2755bb6f`.
- Redacted evidence: `artifacts/M7C/summary.json`, `videos.json`, `comparison.json`, `latents.json`, `models.json`, `hardware.json`, `production-transformer.json`, `production-transformer-regression.json`, `checkpoint-runtime.json`, `port-repair.json`, `tests.json`, ABI/media reports, and empty full-decode error logs. The large playable artifacts remain on-host for coordinator transfer; this worker did not copy them off-host.
- Branch/status at acceptance: source changes were isolated to `codex/ltx-csharp-m7a` through checkpoint `51294b730a211b643118a4013f4b599d57a78de6`. M8 has now integrated that exact checkpoint onto `codex/ltx-csharp`; the redacted heartbeat remains isolated on `codex/ltx-csharp-m7a-status` for this retained worker.

## M7A — accepted checkpoint

- Recovery completed on 2026-08-20 UTC from accepted M6 checkpoint `87ddf3078caf1d41290ee514b6c3c61e06448160`, entirely on isolated branch `codex/ltx-csharp-m7a`; the concurrent M7B source/status branches were not modified.
- Verification: `scripts/remote/verify-milestone.sh M7A` (exit 0). The sole gate rebuilt the Release solution, reran every M1–M6 prerequisite suite, verified the pinned checkpoint files, generated a fresh official Python numerical oracle, and ran the native checkpoint runtime, all 12 real-checkpoint single-GPU pipeline routes, and the standard-profile trainer smoke.
- Plan input: `/root/C_SHARP_PORT_PLAN.M7A.md`, SHA-256 `85da94394a9e37749d579bc333e79eedebe2a61f77324cacd7ecbd5d149247ba`; retained byte-for-byte as `C_SHARP_PORT_PLAN.M7A.md`. Historical accepted M0–M6 plan copies and evidence remain preserved.
- Runtime: native C#/TorchSharp CUDA execution loads the production LTX 2.3.0 monolithic checkpoint lazily, auto-detects its version and 5,947-tensor layout from metadata, runs all 48 AV transformer layers, and implements the convolutional video VAE decoder, audio VAE decoder, BigVGAN-v2 vocoder, and bandwidth extension. The C# runtime made zero Python calls.
- Numerical parity: fixture `m7a-real-checkpoint-runtime-v4` was regenerated from the official Python builders with the portable math SDPA backend. Both transformer modalities matched at every one of 48 block boundaries; final transformer, decoder, and vocoder outputs passed the unchanged BF16 `rtol=2e-2`, `atol=5e-3` gate.
- Pipeline validation: every 12 in-scope module from the accepted M0 inventory executed through `native_csharp_full_checkpoint` and produced validated nonempty MP4/WAV/EXR output. The fixture-only path now requires an explicit `--fixture-mode`; non-fixture CLI execution requires checkpoint and text-embedding inputs and reports its execution provenance.
- Standard-profile training: one full 48-layer backward pass used the unquantized BF16 base checkpoint, batch size 1, rank-2 LoRA, and CUDA AdamW. It retained 21,005,004,544 frozen parameters and 8,448 trainable adapter parameters; loss `0.022675807`, gradient norm `0.14196147`, and update delta norm `0.09013178` were finite and nonzero.
- Earlier-suite recovery: M3 and M4 regenerated CUDA oracle values are compared across GPU architectures at the original FP32 `rtol=1e-4`, `atol=1e-5` threshold while retaining exact structure, integer/inventory fields, and safetensor payloads. The largest observed architecture drift was approximately `1.2e-7`; no numerical criterion was weakened.
- Worker: one NVIDIA A100-SXM4-80GB, compute capability 8.0, 81,920 MiB VRAM. Peak validation usage was 41,847 MiB; minimum available host memory was 247,845 MiB, above the required 8 GiB reserve. Native CUDA remained SASS-only `sm_80` with no PTX.
- Pinned inputs: `ltx-2.3-22b-distilled-1.1.safetensors`, SHA-256 `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`; HDR LoRA, SHA-256 `c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e`; retained scene embeddings, SHA-256 `78bffa6049bae2649a4365ec8769db88052c21348d643e8fc1ce6d483d994c5b`. All sizes and checksums were reverified using only the inherited runtime credential, whose value was never logged or persisted.
- Results: 191 checks passed, 0 failed, 0 skipped: 173 prerequisite checks plus 5 checkpoint component checks, 12 real-checkpoint pipeline routes, and 1 standard-profile trainer smoke.
- Redacted evidence: `artifacts/M7A/summary.json` plus ABI, foundation, storage, core, CUDA, media, low-VRAM, model-download, checkpoint-oracle, checkpoint-runtime, real-pipelines, standard-trainer, and hardware reports. Historical prerequisite/blocker artifacts remain as recovery provenance; all accepted reports record `secrets_recorded=false` where applicable.
- Deferred features remain exactly out of scope: `distilled_mgpu`, `ti2vid_two_stages_mgpu`, and `ti2vid_two_stages_hq_mgpu`; multi-GPU/DDP/FSDP/NCCL/CUDA-IPC/all-to-all including `send_recv_all2all`, `gather_heads`, and `allgather`; B200-only `_fna_kernel` and `_na_kernel`; full fine-tuning; and Hopper-specific native FP8 coverage on this A100.
- Heartbeat isolation remains unchanged: only `codex/ltx-csharp-m7a-status` is updated by this worker. `codex/ltx-csharp-status` remains exclusively owned by the concurrent M7B worker.
- Known failures: none. M7A is accepted and its source checkpoint is an M8 merge parent.

## M6 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M5 checkpoint `829c61c8d3b9733ae9b813bdf7b0dc892e978a7e`; no M6 implementation was started before the pushed documentation/status checkpoint `cba194a0cb94ea19c233bec02c44f4c6126b8e0d`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M6.md`, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`; tracked as `C_SHARP_PORT_PLAN.md` with accepted M0, M1, M2, M3, M4, and M5 plan copies retained as historical evidence. The M6 input is content-identical to the accepted M5 plan and retains the approved M7A/M7B/M8 additions.
- Verification: `scripts/remote/verify-milestone.sh M6` (exit 0). The Release solution build, all M1–M5 prerequisite regressions, deterministic preprocessing runner, and one-step single-GPU low-VRAM LoRA training runner passed.
- Starting/source checkpoint: `cba194a0cb94ea19c233bec02c44f4c6126b8e0d`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Preprocessing scope: deterministic video-latent and caption-connector projection writes paired `video_latents`/`conditions` safetensors with validated geometry, FPS, schema, sample identity, attention mask, exact sample-set pairing, and an overwrite guard. The C# dataset loader consumes those artifacts without retaining preprocessing models on the GPU.
- Training scope: flow-matching interpolation `(1-sigma)*latent + sigma*noise`, velocity target `noise-latent`, masked MSE, BF16 LoRA-only autograd, gradient clipping, and one CPU-offloaded AdamW8bit update passed. Both LoRA A and B gradients and the scalar loss matched the pinned PyTorch `2.13.0+cu132` CUDA oracle.
- Low-VRAM profile: batch size 1, activation chunk size 3 with recomputation, BF16 compute, rowwise-INT8 frozen base weight on CPU, quantized AdamW moments on CPU, and one visible CUDA device. Peak sampled usage was 704 MiB of 32,607 MiB on the existing single RTX 5090 32 GB worker; the fixed miniature acceptance fixture intentionally validates mechanics rather than production-checkpoint capacity.
- Fixed fixture: `m6-low-vram-training-v1`, seed `20260820`, fixture-set SHA-256 `e420c4ae7c248686d2be462aaed9db830db7f7936b6e1704cdda688e9782a471`; tracked manifest SHA-256 `2e95a76187d0f3339e928c7f8a853d7b47eac9fd1336ebc2a0b15bfb1b199cb0`.
- Numerical results: FP32 preprocessing/noise/target checks passed at `rtol=1e-4`, `atol=1e-5`; BF16 loss, gradients, and update passed at `rtol=2e-2`, `atol=5e-3`. The oracle loss was `1.0940678119659424`; C# reported `1.09406793` with gradient norm `0.0490528084`.
- Redacted evidence: `artifacts/M6/abi-smoke.json`, `artifacts/M6/foundation-parity.json`, `artifacts/M6/storage.json`, `artifacts/M6/core-model.json`, `artifacts/M6/cuda-quantization-lora.json`, `artifacts/M6/media-pipelines.json`, `artifacts/M6/low-vram-training.json`, and `artifacts/M6/summary.json`; 173 checks passed, 0 failed, 0 skipped including every prerequisite regression.
- Known failures: none. M7A and M7B are not started; M6 scope ended at this accepted checkpoint.

## M5 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M4 checkpoint `f23816de4e48905904d72bce02924a88dad9705a`; no M5 implementation was started before the pushed documentation/status checkpoint `6185d477520bda9480797f82804f090647629112`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M5.md`, SHA-256 `c4c4cc45fe0a429d70469ca1264aad0ee598026c1cbbffbdaa2234c2594a51a2`; tracked as `C_SHARP_PORT_PLAN.md` with accepted M0, M1, M2, M3, and M4 plan copies retained as historical evidence.
- Verification: `scripts/remote/verify-milestone.sh M5` (exit 0). The Release solution build, all M1–M4 prerequisite regressions, native media bridge, deterministic media/pipeline runner, exact M0 inventory gate, and process-level CLI behavior suite passed.
- Starting/source checkpoint: `6185d477520bda9480797f82804f090647629112`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Pipeline/CLI scope: all 12 in-scope single-GPU modes from the accepted M0 inventory have C# pipeline types, Python-module and command aliases, deterministic seeded execution, help output, success output, and invalid-argument/runtime exit behavior coverage. The only deferred pipeline entries remain exactly `distilled_mgpu`, `ti2vid_two_stages_mgpu`, and `ti2vid_two_stages_hq_mgpu`.
- Media scope: FFmpeg-backed MP4 video and container-audio probe/decode/encode, PCM WAVE I/O, PNG still decode, scene-linear EXR read/write and sequences through the native OpenImageIO bridge, ACEScg/ACEScct/linear HDR conversion, HLG tagging, SDR tone mapping, and HDR IC-LoRA EXR plus ProRes MOV output passed.
- Worker: existing Vast instance `48162892`, single RTX 5090 32 GB, compute capability 12.0. The accepted ABI remains PyTorch `2.13.0+cu132`, CUDA `13.2`, TorchSharp `0.107.0`, and LTX native ABI `1.0`; media dependencies are FFmpeg `6.1.1` and OpenImageIO `2.4.17.0`.
- Fixed fixture: `m5-media-pipelines-v1`, seed `20260820`, fixture-set SHA-256 `f3c19f034acbac41c238f3fb2a15363bd95a7b3a008dde02ce1b35a3c01011e7`; tracked manifest SHA-256 `47a02c81895340ff6755c91c0116ccc14933737ffde4f70ae7aef3e55c8c6a92`.
- Numerical and behavior results: FP32 passed at `rtol=1e-4`, `atol=1e-5`. The M5 suite passed 9 media, 12 pipeline, 12 CLI success, 12 CLI help, and 6 CLI error checks; documented invalid arguments exit 2, runtime media failures exit 1, and successes exit 0 with validated outputs.
- Redacted evidence: `artifacts/M5/abi-smoke.json`, `artifacts/M5/foundation-parity.json`, `artifacts/M5/storage.json`, `artifacts/M5/core-model.json`, `artifacts/M5/cuda-quantization-lora.json`, `artifacts/M5/media-pipelines.json`, and `artifacts/M5/summary.json`; 152 checks passed, 0 failed, 0 skipped including every prerequisite regression.
- Known failures: none. Next milestone: M6 is not started.

## M4 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M3 checkpoint `ebbfd2b38a8eb5fd346ffe9b3e3833bab05277d4`; no M4 implementation was started before the pushed documentation/status checkpoint `4c52084cfbb4926391ae746d566b9dda6728393b`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M4.md`, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`; tracked as `C_SHARP_PORT_PLAN.md` with the accepted M0, M1, M2, and M3 plan copies retained as historical evidence.
- Verification: `scripts/remote/verify-milestone.sh M4` (exit 0). The Release solution build, M1 ABI/native and parity regressions, M2 storage/model-loading regressions, M3 core-model regressions, and the complete M4 CUDA/quantization/LoRA suite passed.
- Starting/source checkpoint: `4c52084cfbb4926391ae746d566b9dda6728393b`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Native scope: all 24 in-scope kernels from the accepted M0 inventory have GPU-tested correctness paths, and all 15 in-scope native bindings have managed test coverage. The five deferred entries remain exactly the three multi-GPU/all-to-all kernels and two B200-only DSL kernels recorded at M0. `libltx_cuda.so` is `sm_120` SASS-only with no PTX.
- Scope completed: stochastic BF16 fused-add, 3D neighborhood attention, fused SwiGLU, SM89/SM90 FP8 GEMM semantics, NVFP4 quantize/dequantize/scaled matrix multiply and diagnostics, FP6 pack/unpack, adjacent and split RMSNorm-RoPE, rowwise INT8, blockwise FP8 quantize/dequantize/GELU/AdaNorm/RMS-FMA, gated attention, and public C# bindings. Architecture-tuned performance is non-gating; the RTX 5090 paths are portable correctness SASS implementations.
- Quantization and LoRA: exact NVFP4 packed payload/block-scale and FP6 fixtures passed; blockwise FP8, rowwise INT8, scaled-matmul, normalization, and fused-op fixtures passed. NVFP4 LoRA fuse passed against the BF16-aggregation oracle and unfuse restored the exact original quantized snapshot; the accepted M2 BF16 LoRA fuse/unfuse regression also passed.
- Worker: existing Vast instance `48162892`, single RTX 5090 32 GB, compute capability 12.0. The accepted ABI remains PyTorch `2.13.0+cu132`, CUDA `13.2`, TorchSharp `0.107.0`, and LTX native ABI `1.0`.
- Fixed fixture: `m4-cuda-quantization-lora-v1`, seed `20260820`, fixture-set SHA-256 `420f7a7f17e69898f0f4b43a01ebb2d2ddedf847f1393a651445738e7e2745b0`; tracked manifest SHA-256 `fda7db0808a76368ca396a961456ed8dd2719914104646cd5be28eb6b33a4f1b`.
- Numerical results: FP32 passed at `rtol=1e-4`, `atol=1e-5`; BF16/FP8 passed at `rtol=2e-2`, `atol=5e-3`. The M4 suite passed 24 kernel, 15 binding, 4 quantization, and 2 LoRA checks.
- Redacted evidence: `artifacts/M4/abi-smoke.json`, `artifacts/M4/foundation-parity.json`, `artifacts/M4/storage.json`, `artifacts/M4/core-model.json`, `artifacts/M4/cuda-quantization-lora.json`, and `artifacts/M4/summary.json`; 101 checks passed, 0 failed, 0 skipped including all prerequisite regressions.
- Known failures: none. Next milestone: M5 is not started.

## M3 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M2 checkpoint `f53c216a4a9598d1bc626923744095aac7311a46`; no M3 implementation was started before the pushed documentation/status checkpoint `16531ddd4cd279232c021a4c939d71b46a042535`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M3.md`, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`; tracked as `C_SHARP_PORT_PLAN.md` with the accepted M0, M1, and M2 plan copies retained as historical evidence.
- Verification: `scripts/remote/verify-milestone.sh M3` (exit 0). The Release solution build, M1 ABI/native and parity regressions, M2 storage/model-loading regressions, and all five required M3 core-model fixture suites passed.
- Starting/source checkpoint: `16531ddd4cd279232c021a4c939d71b46a042535`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Scope completed: transformer timestep embeddings, tanh-GELU feed-forward, and masked/unmasked multi-head attention; video VAE patch/unpatch, pixel normalization, and latent channel statistics; audio Snake/SnakeBeta and vocoder layout/final activation; conditioning cross/self-attention masks; and deterministic no-offload, CPU-offload, and safetensors disk-offload linear execution.
- Worker: existing Vast instance `48162892`, single RTX 5090 32 GB, compute capability 12.0. The accepted ABI remains PyTorch `2.13.0+cu132`, CUDA `13.2`, TorchSharp `0.107.0`, and `sm_120` SASS-only native code.
- Fixed fixture: `m3-core-model-execution-v1`, seed `20260820`, fixture-set SHA-256 `6964b807c01b0b15a191031432eaf54b4a6945d3cfa96b27cdddda47a204c1cd`. Individual file SHA-256 values are recorded in `artifacts/M3/summary.json`; no production model checkpoint bytes were required for M3.
- Numerical results: FP32 passed at `rtol=1e-4`, `atol=1e-5`; BF16 passed at `rtol=2e-2`, `atol=5e-3`. The new suites passed transformer 7, VAE 8, audio/vocoder 9, conditioning 4, and offload 6 checks.
- Redacted evidence: `artifacts/M3/abi-smoke.json`, `artifacts/M3/foundation-parity.json`, `artifacts/M3/storage.json`, `artifacts/M3/core-model.json`, and `artifacts/M3/summary.json`; 56 checks passed, 0 failed, 0 skipped including all prerequisite regressions.
- Known failures: none. Next milestone: M4 is not started.

## M2 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M1 checkpoint `e9dcbdf69c658ad026bdec6ee5cd4642847215a8`; no M2 implementation was started before the pushed documentation/status checkpoint `67db844be650eeb2be35d4a8849464f58a65ecd9`.
- Plan input: `/root/C_SHARP_PORT_PLAN.M2.md`, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`; tracked as `C_SHARP_PORT_PLAN.md` with accepted M0 and M1 copies retained as historical evidence.
- Verification: `scripts/remote/verify-milestone.sh M2` (exit 0). The Release solution build, M1 ABI/native and parity regressions, exact safetensors/LoRA round trips, and fixed sharded checkpoint-loader fixtures all passed.
- Starting/source checkpoint: `67db844be650eeb2be35d4a8849464f58a65ecd9`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Scope completed: schema-validated safetensors read/write with deterministic headers and raw dtype preservation; TorchSharp tensor conversion; metadata-only reads and JSON metadata parsing; sharded checkpoint load/filter/key mapping; LoRA loading, Comfy prefix mapping, and BF16-aggregation fuse/unfuse.
- Worker: existing Vast instance `48162892`, single RTX 5090 32 GB, compute capability 12.0. The accepted M1 ABI remained PyTorch `2.13.0+cu132`, CUDA `13.2`, TorchSharp `0.107.0`, and `sm_120` SASS-only native code.
- Fixed fixture: `m2-storage-model-loading-v1`, fixture-set SHA-256 `4a8dc2680c3adf8acb78f5ff7a4ac60001eedeb9c3638848551f334c601448bf`. Individual file SHA-256 values are recorded in `artifacts/M2/summary.json`; no model checkpoint bytes were required for M2.
- Numerical results: FP32 passed at `rtol=1e-4`, `atol=1e-5`; BF16 passed at `rtol=2e-2`, `atol=5e-3`. Safetensors and LoRA round trips preserved dtype, shape, payload, and metadata exactly in both C# and the pinned Python safetensors `0.6.2` verifier.
- Redacted evidence: `artifacts/M2/abi-smoke.json`, `artifacts/M2/foundation-parity.json`, `artifacts/M2/storage.json`, and `artifacts/M2/summary.json`; 22 checks passed, 0 failed, 0 skipped.
- A first verification attempt exposed randomized metadata-key ordering in the Python reference writer; fixture generation was canonicalized, reproduced byte-for-byte across clean runs, and the complete unchanged gate then passed twice. Known failures: none. Next milestone: M3 is not started.

## M1 — accepted checkpoint

- Handoff acknowledged on 2026-08-20 UTC from accepted M0 checkpoint `28ad7ad1e6e1594e5f3b8ce1a29d7065fb90bacb`; no M1 implementation was started before this documentation/status checkpoint.
- Verification: `scripts/remote/verify-milestone.sh M1` (exit 0). The Release solution build completed with 0 warnings and 0 errors; the ABI/native smoke and parity runner both passed.
- Starting/source checkpoint: `95328d650b26d7a6fb17f6ca991fb9534ddc1e90`; the final fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push because it cannot be embedded in its own commit.
- Scope completed: .NET 10 solution and project boundaries for `Ltx.Core`, `Ltx.Cuda`, `Ltx.SafeTensors`, `Ltx.Media`, `Ltx.Pipelines`, `Ltx.Trainer`, and `Ltx.Cli`; M2+ assemblies remain explicit foundation-only placeholders.
- ABI: TorchSharp managed API `0.107.0`; fork source `8f4def03b641b6753f18076aa5438f8eaaef2d30`; PyTorch `2.13.0+cu132`; CUDA `13.2`; C++11 ABI enabled; LTX native ABI `1.0`. No stock TorchSharp CUDA runtime package is referenced.
- Worker: existing Vast instance `48162892`, single RTX 5090 32 GB, compute capability 12.0. `libltx_cuda.so` contains `sm_120` SASS and no PTX.
- Numerical fixture: `m1-foundation-v1`, SHA-256 `84199de9aaac9d64d6c9812e894a87bb11352faa137024058b9b42d4b4b47ee2`; 2 FP32 cases passed at `rtol=1e-4`, `atol=1e-5` against the pinned Python oracle.
- Redacted evidence: `artifacts/M1/abi-smoke.json`, `artifacts/M1/parity.json`, and `artifacts/M1/summary.json`; 6 checks passed, 0 failed, 0 skipped.
- Compatibility note: PyTorch 2.13 removed named tensors. The maintained TorchSharp fork preserves the managed entry points and returns an explicit unsupported error if called; no in-scope LTX M1 path uses named tensors.
- Known failures: none. Next milestone: M2 is not started.

## M0 — accepted checkpoint

- Verification: `scripts/remote/verify-milestone.sh M0` (exit 0).
- Starting/source reference: `400fd31054597515f47125691032c04b1c3ee24e`; fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push (it cannot be embedded in its own commit).
- Plan: `C_SHARP_PORT_PLAN.M0.md`, SHA-256 `d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6`.
- Worker: Vast instance `48162892`, single RTX 5090 32 GB, `$0.4259259259/hour`, direct SSH, 32 GB nominal RAM, 200 GiB disk. CUDA 13.2 / driver 580.95.05 minor-version compatibility; compute capability 12.0; native smoke contains `sm_120` SASS and no PTX.
- Approved limits enforced: routine worker at or below `$0.60/hour`; H100 validation worker at or below `$1.20/hour`; per-instance duration and total spend are unlimited. No H100 was created for M0.
- Remote access: ephemeral ED25519 deploy key fingerprint `SHA256:Vvy5vbMRoLgmo98uwFk8ZpTBhaX5VgmwE4CEPx++JP4`; write-enabled push proven. The registration ID is manager-held/redacted because the rental-host GitHub token is intentionally denied deploy-key administration (HTTP 403); recovery maps this fingerprint to the ID before revocation. No private key or credential is tracked.
- Codex CLI 0.148.0 is signed in with ChatGPT; no OpenAI API key is stored. The redacted `codex/ltx-csharp-status` branch contains only `heartbeat.json` and was published successfully.
- Hugging Face: fine-grained read-only token presence was detected only in PID 1 and used transiently for 16 authenticated metadata probes; no token value or model bytes were logged/downloaded. Required disk with cache/fixture reserve and 20% headroom is `124242028445` bytes, within the disk gate.
- GHCR: anonymous manifest access to `ghcr.io/miloszkukla/ltx-csharp-base` returned `HTTP 401 Unauthorized` on this fresh Vast worker. The accepted strategy is the checked-in `nvidia/cuda@sha256:293837d7ad950f195e6ba5e3c0478c50b8c80349a70ad14687f39ed8ff0a8e4a` Linux/amd64 base plus `scripts/remote/bootstrap-host.sh`; the `packages: write`/`GITHUB_TOKEN` publish workflow remains checked in and uses no personal token.
- Source surface: 39 Python packages, 280 production modules, 26 CLI entry points, 194 flag declarations, 15 pipelines, 47 native sources, 29 kernels, and 15 native bindings; multi-GPU/all-to-all and B200-only DSL entries are explicitly deferred.
- Smoke tests: TorchSharp tensor, FFmpeg/OpenImageIO/OpenCV P/Invoke, CUDA native kernel, and SASS-only checks all passed. Evidence: `artifacts/M0/summary.json`.
- Known supported fallback: GHCR is not anonymously pullable, as recorded above. No uncommitted source work may remain after the checkpoint.
- Next milestone: M1 is not started.

## Required model/checkpoint pins

Repository revisions:

- `Lightricks/LTX-2.5@6c7e5e573ac1667efc83407806fe9b0b93730e60`
- `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler@74c4e68ee7dd99f3997d5a1bb1a3784941822222`
- `Lightricks/LTX-2.3@6b5a83e3045eaf8e46cfa0acce512412aa2b9cce`
- `Lightricks/LTX-2.3-22b-IC-LoRA-HDR@577ab50f447358d00ba68ef204648dfe05646300`
- `Lightricks/LTX-2.3-22b-IC-LoRA-DubIt@1456334c3d69924de5083e553733b108ed1147f2`
- `google/gemma-3-12b-it-qat-q4_0-unquantized@68f7ee4fbd59087436ada77ed2d62f373fdd4482`

Required LFS SHA-256 values (paths, sizes, and checksum source are in `artifacts/M0/model-access.json`):

- `ltx25-dev-transformer`: `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584`
- `ltx25-distilled-transformer`: `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4`
- `ltx25-text-encoder`: `ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1`
- `ltx25-video-vae-diffusion`: `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3`
- `ltx25-video-vae-convolutional`: `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d`
- `ltx25-audio-vae`: `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5`
- `ltx25-spatial-upscaler`: `eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8`
- `ltx25-temporal-upscaler`: `2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe`
- `ltx25-distilled-lora`: `86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5`
- `ltx25-duration-head`: `2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190`
- `ltx25-detailing-ic-lora`: `984851b769ea2bcb4c9e0a239a7676239e42c6a6001ddc69943b41ff0b283c1d`
- `ltx23-distilled-checkpoint`: `b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`
- `ltx23-spatial-upscaler`: `5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed`
- `ltx23-gemma3-assets` shards: `e6fb899db428481aafb45a20130457df6e247e7cb03b7d9f01ee4bc2a9a08138`, `d251e7fe9799d529405ddb61705a44cd700bd30a8b66a8d44ae26ddf8365dbc6`, `0684ef801385f0669a0b3e4ab160c50877efdbfa40eb97788595985de2743e78`, `b4b964e6526f81ccfa625c900b72ce92d5e0fd2debb75998763038ad06b9c541`, `4ef2de8f93e165b4e02425769fc566000b0674256ef0c3a27b23a0d45eb12088`
- `ltx23-hdr-assets`: `c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e`, `78bffa6049bae2649a4365ec8769db88052c21348d643e8fc1ce6d483d994c5b`
- `ltx23-dubit-lora`: `fc415b12cb639e78511bc264f85080c2f7b188e334c1d9fade76b310e2bc419c`

M0 has no numerical fixture revision; numerical parity fixtures begin in M1.
