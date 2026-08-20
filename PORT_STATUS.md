# C# port status

## M0 — accepted checkpoint

- Verification: `scripts/remote/verify-milestone.sh M0` (exit 0).
- Starting/source reference: `400fd31054597515f47125691032c04b1c3ee24e`; fork checkpoint SHA is recorded by the redacted heartbeat immediately after the atomic push (it cannot be embedded in its own commit).
- Plan: `C_SHARP_PORT_PLAN.md`, SHA-256 `d5b0f9559011a690da8e59e20455b2290ac882a85281f5f8640e2916ac7bbeb6`.
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
