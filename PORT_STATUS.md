# C# port status

## Plan revision history

- M0 bootstrap plan: `C_SHARP_PORT_PLAN.M0.md`, SHA-256 `d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6`. This is the byte-for-byte plan accepted at M0 and remains immutable historical evidence.
- Accepted M1 plan: `C_SHARP_PORT_PLAN.M1.md`, revision M1, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `e9dcbdf69c658ad026bdec6ee5cd4642847215a8`.
- Accepted M2 plan: `C_SHARP_PORT_PLAN.M2.md`, revision M2, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `f53c216a4a9598d1bc626923744095aac7311a46`.
- Accepted M3 plan: `C_SHARP_PORT_PLAN.M3.md`, revision M3, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. This preserves the byte-for-byte plan accepted with checkpoint `ebbfd2b38a8eb5fd346ffe9b3e3833bab05277d4`.
- Current M4 plan: `C_SHARP_PORT_PLAN.md`, revision M4, SHA-256 `7dbc170e428101452a8764ba6fd2b3500b3b7db0397df43d44bda55a09a8f9ea`. It byte-matches the coordinator handoff at `/root/C_SHARP_PORT_PLAN.M4.md`; the coordinator's M4 revision is content-identical to the accepted M1, M2, and M3 revisions.

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
