using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public sealed record CheckpointTransformerInput(
    Tensor Latent,
    Tensor Context,
    Tensor? ContextMask,
    Tensor Timesteps,
    Tensor Sigma,
    Tensor Positions,
    bool Enabled = true);

public sealed record CheckpointTransformerOutput(Tensor? Video, Tensor? Audio) : IDisposable
{
    public void Dispose()
    {
        Video?.Dispose();
        Audio?.Dispose();
    }
}

/// <summary>A trainable LoRA projection attached to one production checkpoint linear layer.</summary>
public sealed record CheckpointLoraAdapter(Tensor A, Tensor B, double Scale)
{
    public int Rank => checked((int)A.shape[0]);
}

/// <summary>
/// Executes the production LTX-2.3/2.5 audio-video transformer directly from safetensors weights.
/// The implementation intentionally uses only TorchSharp tensor operations; Python is not part of
/// the runtime path. Checkpoint tensors are loaded lazily and cached on the selected CUDA device.
/// </summary>
public sealed class CheckpointTransformerRuntime : IDisposable
{
    private const string Root = "model.diffusion_model.";
    private readonly CheckpointTensorStore store;
    private readonly Device device;
    private readonly TransformerConfig config;
    private readonly IReadOnlyDictionary<string, CheckpointLoraAdapter> adapters;
    private readonly bool cacheWeights;
    private bool disposed;

    public CheckpointTransformerRuntime(
        CheckpointTensorStore store,
        Device device,
        IReadOnlyDictionary<string, CheckpointLoraAdapter>? adapters = null,
        bool cacheWeights = true)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(device);
        this.store = store;
        this.device = device;
        this.adapters = adapters ?? new Dictionary<string, CheckpointLoraAdapter>(StringComparer.Ordinal);
        this.cacheWeights = cacheWeights;
        config = TransformerConfig.Parse(store.Metadata);
        ValidateCheckpointSurface();
        ValidateAdapters();
    }

    public int LayerCount => config.LayerCount;

    public string ModelVersion => config.ModelVersion;

    public CheckpointTransformerOutput Forward(
        CheckpointTransformerInput? video,
        CheckpointTransformerInput? audio,
        Action<int, Tensor?, Tensor?>? observeBlock = null,
        Action<string, Tensor>? observeTensor = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (video is null && audio is null)
        {
            throw new ArgumentException("At least one transformer modality is required.");
        }
        ValidateInput(video, config.VideoDimension, 3, nameof(video));
        ValidateInput(audio, config.AudioDimension, 1, nameof(audio));

        using var forwardScope = NewDisposeScope();
        var videoState = video is null ? null : Prepare(video, isAudio: false, observeTensor);
        var audioState = audio is null ? null : Prepare(audio, isAudio: true, observeTensor);
        if (videoState is not null && audioState is not null)
        {
            var unresolvedVideo = videoState;
            var unresolvedAudio = audioState;
            videoState = PrepareCrossState(unresolvedVideo, unresolvedAudio, isAudio: false);
            audioState = PrepareCrossState(unresolvedAudio, unresolvedVideo, isAudio: true);
        }
        observeBlock?.Invoke(-1, videoState?.X, audioState?.X);

        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            using var blockScope = NewDisposeScope();
            var previousVideo = videoState?.X;
            var previousAudio = audioState?.X;
            (videoState, audioState) = ExecuteBlock(layer, videoState, audioState, observeTensor);
            observeBlock?.Invoke(layer, videoState?.X, audioState?.X);
            if (videoState is not null && !ReferenceEquals(previousVideo, videoState.X))
            {
                videoState = videoState with { X = videoState.X.MoveToOuterDisposeScope() };
                previousVideo?.Dispose();
            }
            if (audioState is not null && !ReferenceEquals(previousAudio, audioState.X))
            {
                audioState = audioState with { X = audioState.X.MoveToOuterDisposeScope() };
                previousAudio?.Dispose();
            }
        }

        var videoOutput = videoState is null
            ? null
            : ProcessOutput(
                videoState.X,
                videoState.EmbeddedTimestep,
                Root + "scale_shift_table",
                Root + "proj_out",
                observeTensor,
                "video");
        var audioOutput = audioState is null
            ? null
            : ProcessOutput(
                audioState.X,
                audioState.EmbeddedTimestep,
                Root + "audio_scale_shift_table",
                Root + "audio_proj_out",
                observeTensor,
                "audio");
        return new CheckpointTransformerOutput(
            videoOutput?.MoveToOuterDisposeScope(),
            audioOutput?.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Runs the checkpoint's eight-layer embeddings connector. Inputs are the video/audio
    /// feature projections emitted by the matching Gemma feature extractor.
    /// </summary>
    public Tensor ConnectTextFeatures(Tensor features, Tensor binaryMask, bool isAudio)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(binaryMask);
        ObjectDisposedException.ThrowIf(disposed, this);
        var dimension = isAudio ? config.AudioDimension : config.VideoDimension;
        if (features.dim() != 3 || features.shape[2] != dimension || binaryMask.dim() != 2 ||
            binaryMask.shape[0] != features.shape[0] || binaryMask.shape[1] != features.shape[1])
        {
            throw new ArgumentException("Text feature and mask shapes do not match the checkpoint connector.");
        }
        if (features.shape[1] % config.ConnectorRegisters != 0)
        {
            throw new ArgumentException(
                $"Connector sequence length must be divisible by {config.ConnectorRegisters}.", nameof(features));
        }

        using var scope = NewDisposeScope();
        var prefix = Root + (isAudio ? "audio_embeddings_connector." : "video_embeddings_connector.");
        var x = features.to(config.ComputeDType, device, non_blocking: false);
        var mask = binaryMask.to(config.ComputeDType, device, non_blocking: false);
        var registers = Weight(prefix + "learnable_registers")
            .repeat([features.shape[1] / config.ConnectorRegisters, 1])
            .unsqueeze(0)
            .expand(features.shape[0], -1, -1);
        x = TorchSharpRuntime.Add(
            mask.unsqueeze(-1).mul(x),
            mask.unsqueeze(-1).mul(-1).add(1).mul(registers));
        var positions = arange(0, x.shape[1], dtype: ScalarType.Float32, device: device)
            .reshape(1, 1, x.shape[1])
            .expand(x.shape[0], -1, -1);
        var rope = PrepareRope(positions, dimension, [config.ConnectorMaximumPosition], config.ConnectorHeads);

        for (var layer = 0; layer < config.ConnectorLayers; layer++)
        {
            var block = $"{prefix}transformer_1d_blocks.{layer}.";
            var normalized = RmsNorm(x);
            var attended = Attention(normalized, null, null, rope, null, block + "attn1", config.ConnectorHeads);
            x = TorchSharpRuntime.Add(x, attended);
            normalized = RmsNorm(x);
            x = TorchSharpRuntime.Add(x, FeedForward(normalized, block + "ff"));
        }
        return RmsNorm(x).MoveToOuterDisposeScope();
    }

    public void Dispose()
    {
        disposed = true;
    }

    private TransformerState Prepare(
        CheckpointTransformerInput input,
        bool isAudio,
        Action<string, Tensor>? observeTensor)
    {
        var dimension = isAudio ? config.AudioDimension : config.VideoDimension;
        var heads = isAudio ? config.AudioHeads : config.VideoHeads;
        var maximumPositions = isAudio ? config.AudioMaximumPositions : config.VideoMaximumPositions;
        var prefix = Root + (isAudio ? "audio_" : string.Empty);
        var latent = input.Latent.to(config.ComputeDType, device, non_blocking: false);
        // Scheduler values remain FP32 through the sinusoidal embedding. Quantizing sigma
        // before the checkpoint's 1000x scale changes its high-frequency phases materially.
        var sourceTimesteps = input.Timesteps.to(ScalarType.Float32, device, non_blocking: false);
        var sourceSigma = input.Sigma.to(ScalarType.Float32, device, non_blocking: false);
        var x = Linear(latent, prefix + "patchify_proj");
        var (timestep, embedded) = Ada(
            sourceTimesteps.mul(config.TimestepScale),
            prefix + "adaln_single",
            dimension,
            9,
            observeTensor,
            (isAudio ? "audio" : "video") + "_adaln");
        var (promptTimestep, _) = Ada(
            sourceSigma.mul(config.TimestepScale),
            prefix + "prompt_adaln_single",
            dimension,
            2);
        var modalityName = isAudio ? "audio" : "video";
        observeTensor?.Invoke($"{modalityName}_prepared", x);
        observeTensor?.Invoke($"{modalityName}_embedded_timestep", embedded);
        observeTensor?.Invoke($"{modalityName}_timestep", timestep);
        observeTensor?.Invoke($"{modalityName}_prompt_timestep", promptTimestep);
        var context = input.Context.to(config.ComputeDType, device, non_blocking: false);
        var mask = input.ContextMask?.to(config.ComputeDType, device, non_blocking: false);
        var positions = input.Positions.to(ScalarType.Float32, device, non_blocking: false);
        var rope = PrepareRope(positions, dimension, maximumPositions, heads);
        return new TransformerState(
            x,
            context,
            mask,
            timestep,
            embedded,
            promptTimestep,
            rope,
            null,
            null,
            null,
            sourceTimesteps,
            sourceSigma,
            input.Enabled);
    }

    private (TransformerState? Video, TransformerState? Audio) ExecuteBlock(
        int layer,
        TransformerState? video,
        TransformerState? audio,
        Action<string, Tensor>? observeTensor)
    {
        var block = $"{Root}transformer_blocks.{layer}.";
        var vx = video?.X;
        var ax = audio?.X;
        var runVideo = video is { Enabled: true } && vx!.numel() > 0;
        var runAudio = audio is { Enabled: true } && ax!.numel() > 0;

        if (runVideo)
        {
            var values = AdaValues(block + "scale_shift_table", video!.Timestep, 0, 3);
            var normalized = ModulatedRmsNorm(vx!, values[1], values[0]);
            var attended = Attention(
                normalized,
                null,
                null,
                video.Rope,
                null,
                block + "attn1",
                config.VideoHeads,
                observeTensor);
            vx = TorchSharpRuntime.Add(vx!, attended.mul(values[2]));
            var afterSelfAttention = RmsNorm(vx);
            vx = TorchSharpRuntime.Add(vx, TextCrossAttention(
                afterSelfAttention,
                video.Context,
                video.ContextMask,
                video.Timestep,
                video.PromptTimestep,
                block + "scale_shift_table",
                block + "prompt_scale_shift_table",
                block + "attn2",
                config.VideoHeads,
                observeTensor));
        }

        if (runAudio)
        {
            var values = AdaValues(block + "audio_scale_shift_table", audio!.Timestep, 0, 3);
            var normalized = ModulatedRmsNorm(ax!, values[1], values[0]);
            var attended = Attention(
                normalized,
                null,
                null,
                audio.Rope,
                null,
                block + "audio_attn1",
                config.AudioHeads,
                observeTensor);
            ax = TorchSharpRuntime.Add(ax!, attended.mul(values[2]));
            var afterSelfAttention = RmsNorm(ax);
            ax = TorchSharpRuntime.Add(ax, TextCrossAttention(
                afterSelfAttention,
                audio.Context,
                audio.ContextMask,
                audio.Timestep,
                audio.PromptTimestep,
                block + "audio_scale_shift_table",
                block + "audio_prompt_scale_shift_table",
                block + "audio_attn2",
                config.AudioHeads,
                observeTensor));
        }

        if (video is not null && audio is not null && (runVideo || runAudio))
        {
            var videoBeforeCross = vx!;
            var audioBeforeCross = ax!;
            var preparedVideo = video;
            var preparedAudio = audio;
            if (runVideo && audioBeforeCross.numel() > 0)
            {
                var videoValues = CrossAdaValues(
                    block + "scale_shift_table_a2v_ca_video",
                    preparedVideo.CrossScaleShiftTimestep!,
                    preparedVideo.CrossGateTimestep!,
                    0);
                var audioValues = CrossAdaValues(
                    block + "scale_shift_table_a2v_ca_audio",
                    preparedAudio.CrossScaleShiftTimestep!,
                    preparedAudio.CrossGateTimestep!,
                    0);
                var q = ModulatedRmsNorm(videoBeforeCross, videoValues.Scale, videoValues.Shift);
                var kv = ModulatedRmsNorm(audioBeforeCross, audioValues.Scale, audioValues.Shift);
                vx = TorchSharpRuntime.Add(videoBeforeCross, Attention(
                    q,
                    kv,
                    null,
                    preparedVideo.CrossRope,
                    preparedAudio.CrossRope,
                    block + "audio_to_video_attn",
                    config.AudioHeads,
                    observeTensor).mul(videoValues.Gate));
            }
            if (runAudio && videoBeforeCross.numel() > 0)
            {
                var audioValues = CrossAdaValues(
                    block + "scale_shift_table_a2v_ca_audio",
                    preparedAudio.CrossScaleShiftTimestep!,
                    preparedAudio.CrossGateTimestep!,
                    2);
                var videoValues = CrossAdaValues(
                    block + "scale_shift_table_a2v_ca_video",
                    preparedVideo.CrossScaleShiftTimestep!,
                    preparedVideo.CrossGateTimestep!,
                    2);
                var q = ModulatedRmsNorm(audioBeforeCross, audioValues.Scale, audioValues.Shift);
                var kv = ModulatedRmsNorm(videoBeforeCross, videoValues.Scale, videoValues.Shift);
                ax = TorchSharpRuntime.Add(audioBeforeCross, Attention(
                    q,
                    kv,
                    null,
                    preparedAudio.CrossRope,
                    preparedVideo.CrossRope,
                    block + "video_to_audio_attn",
                    config.AudioHeads,
                    observeTensor).mul(audioValues.Gate));
            }
        }

        if (runVideo)
        {
            var values = AdaValues(block + "scale_shift_table", video!.Timestep, 3, 3);
            var normalized = ModulatedRmsNorm(vx!, values[1], values[0]);
            vx = TorchSharpRuntime.Add(vx!, FeedForward(normalized, block + "ff", observeTensor).mul(values[2]));
        }
        if (runAudio)
        {
            var values = AdaValues(block + "audio_scale_shift_table", audio!.Timestep, 3, 3);
            var normalized = ModulatedRmsNorm(ax!, values[1], values[0]);
            ax = TorchSharpRuntime.Add(ax!, FeedForward(normalized, block + "audio_ff", observeTensor).mul(values[2]));
        }
        return (video is null ? null : video with { X = vx! }, audio is null ? null : audio with { X = ax! });
    }

    private TransformerState PrepareCrossState(
        TransformerState state,
        TransformerState other,
        bool isAudio)
    {
        if (state.CrossRope is not null)
        {
            return state;
        }
        var dimension = isAudio ? config.AudioDimension : config.VideoDimension;
        var modalityName = isAudio ? "audio" : "video";
        var positions = state.Rope.SourcePositions.narrow(1, 0, 1);
        var crossRope = PrepareRope(
            positions,
            config.AudioDimension,
            [Math.Max(config.VideoMaximumPositions[0], config.AudioMaximumPositions[0])],
            config.AudioHeads);
        var (scaleShift, _) = Ada(
            state.SourceTimesteps.mul(config.TimestepScale),
            $"{Root}av_ca_{modalityName}_scale_shift_adaln_single",
            dimension,
            4);
        var gatePrefix = isAudio ? "av_ca_v2a_gate_adaln_single" : "av_ca_a2v_gate_adaln_single";
        var (gate, _) = Ada(
            other.SourceSigma.mul(config.AudioVideoTimestepScale),
            Root + gatePrefix,
            dimension,
            1);
        return state with
        {
            CrossRope = crossRope,
            CrossScaleShiftTimestep = scaleShift,
            CrossGateTimestep = gate,
        };
    }

    private Tensor TextCrossAttention(
        Tensor normalized,
        Tensor context,
        Tensor? contextMask,
        Tensor timestep,
        Tensor promptTimestep,
        string scaleShiftTable,
        string promptScaleShiftTable,
        string attentionPrefix,
        int heads,
        Action<string, Tensor>? observeTensor)
    {
        var queryValues = AdaValues(scaleShiftTable, timestep, 6, 3);
        var promptTable = Weight(promptScaleShiftTable).unsqueeze(0).unsqueeze(0);
        var prompt = promptTimestep.reshape(promptTimestep.shape[0], promptTimestep.shape[1], 2, -1);
        var modulation = TorchSharpRuntime.Add(promptTable, prompt);
        var shift = modulation.narrow(2, 0, 1).squeeze(2);
        var scale = modulation.narrow(2, 1, 1).squeeze(2);
        var q = TorchSharpRuntime.Add(normalized.mul(queryValues[1].add(1)), queryValues[0]);
        var kv = TorchSharpRuntime.Add(context.mul(scale.add(1)), shift);
        return Attention(q, kv, contextMask, null, null, attentionPrefix, heads, observeTensor).mul(queryValues[2]);
    }

    private CrossAda CrossAdaValues(string tableName, Tensor scaleShift, Tensor gate, int start)
    {
        var table = Weight(tableName);
        var scaleShiftValues = TorchSharpRuntime.Add(
            table.narrow(0, 0, 4).unsqueeze(0).unsqueeze(0),
            scaleShift.reshape(scaleShift.shape[0], scaleShift.shape[1], 4, -1));
        var gateValues = TorchSharpRuntime.Add(
            table.narrow(0, 4, 1).unsqueeze(0).unsqueeze(0),
            gate.reshape(gate.shape[0], gate.shape[1], 1, -1));
        return new CrossAda(
            scaleShiftValues.narrow(2, start, 1).squeeze(2),
            scaleShiftValues.narrow(2, start + 1, 1).squeeze(2),
            gateValues.squeeze(2));
    }

    private Tensor[] AdaValues(string tableName, Tensor timestep, int start, int count)
    {
        var table = Weight(tableName);
        var values = TorchSharpRuntime.Add(
            table.unsqueeze(0).unsqueeze(0),
            timestep.reshape(timestep.shape[0], timestep.shape[1], table.shape[0], -1));
        return Enumerable.Range(start, count)
            .Select(index => values.narrow(2, index, 1).squeeze(2))
            .ToArray();
    }

    private (Tensor Modulation, Tensor Embedded) Ada(
        Tensor input,
        string prefix,
        int dimension,
        int coefficient,
        Action<string, Tensor>? observeTensor = null,
        string? observationPrefix = null)
    {
        var flattened = input.flatten();
        var projected = TransformerExecution.TimestepEmbedding(
            flattened,
            256,
            flipSinToCos: true,
            downscaleFrequencyShift: 0);
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_time_projection", projected);
        }
        var embedded = Linear(projected.to(config.ComputeDType), prefix + ".emb.timestep_embedder.linear_1");
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_linear_1", embedded);
        }
        embedded = Silu(embedded);
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_embed_silu", embedded);
        }
        embedded = Linear(embedded, prefix + ".emb.timestep_embedder.linear_2");
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_embedded", embedded);
        }
        var activated = Silu(embedded);
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_silu", activated);
        }
        var modulation = Linear(activated, prefix + ".linear");
        if (observationPrefix is not null)
        {
            observeTensor?.Invoke(observationPrefix + "_modulation", modulation);
        }
        return (
            modulation.reshape(input.shape[0], -1, coefficient * dimension),
            embedded.reshape(input.shape[0], -1, dimension));
    }

    private Tensor ProcessOutput(
        Tensor x,
        Tensor embeddedTimestep,
        string tableName,
        string projectionPrefix,
        Action<string, Tensor>? observeTensor,
        string observationPrefix)
    {
        var table = Weight(tableName).unsqueeze(0).unsqueeze(0);
        var values = TorchSharpRuntime.Add(table, embeddedTimestep.unsqueeze(2));
        var shift = values.narrow(2, 0, 1).squeeze(2);
        var scale = values.narrow(2, 1, 1).squeeze(2);
        var normalized = nn.functional.layer_norm(x, [x.shape[^1]], eps: 1e-6);
        observeTensor?.Invoke(observationPrefix + "_output_norm", normalized);
        var projectionInput = TorchSharpRuntime.Add(normalized.mul(scale.add(1)), shift);
        observeTensor?.Invoke(observationPrefix + "_output_projection_input", projectionInput);
        return Linear(projectionInput, projectionPrefix);
    }

    private Tensor Attention(
        Tensor x,
        Tensor? context,
        Tensor? binaryMask,
        Rope? queryRope,
        Rope? keyRope,
        string prefix,
        int heads,
        Action<string, Tensor>? observeTensor = null)
    {
        context ??= x;
        observeTensor?.Invoke(prefix + ".input", x);
        observeTensor?.Invoke(prefix + ".context", context);
        var rawQ = Linear(x, prefix + ".to_q");
        var rawK = Linear(context, prefix + ".to_k");
        observeTensor?.Invoke(prefix + ".to_q", rawQ);
        observeTensor?.Invoke(prefix + ".to_k", rawK);
        var q = RmsNorm(rawQ, Weight(prefix + ".q_norm.weight"));
        var k = RmsNorm(rawK, Weight(prefix + ".k_norm.weight"));
        observeTensor?.Invoke(prefix + ".q_norm", q);
        observeTensor?.Invoke(prefix + ".k_norm", k);
        var v = Linear(context, prefix + ".to_v");
        observeTensor?.Invoke(prefix + ".to_v", v);
        if (queryRope is not null)
        {
            q = ApplyRope(q, queryRope);
            k = ApplyRope(k, keyRope ?? queryRope);
            observeTensor?.Invoke(prefix + ".q_rope", q);
            observeTensor?.Invoke(prefix + ".k_rope", k);
        }

        var batch = q.shape[0];
        var queryTokens = q.shape[1];
        var keyTokens = k.shape[1];
        var headDimension = q.shape[2] / heads;
        var qh = q.reshape(batch, queryTokens, heads, headDimension).transpose(1, 2);
        var kh = k.reshape(batch, keyTokens, heads, headDimension).transpose(1, 2);
        var vh = v.reshape(batch, keyTokens, heads, headDimension).transpose(1, 2);
        Tensor? additiveMask = null;
        if (binaryMask is not null)
        {
            var mask = binaryMask.dim() switch
            {
                2 => binaryMask.unsqueeze(1).unsqueeze(1),
                3 => binaryMask.unsqueeze(1),
                4 => binaryMask,
                _ => throw new ArgumentException("Attention mask must have rank two, three, or four."),
            };
            additiveMask = mask.to(qh).sub(1).mul(10_000);
        }
        var attended = nn.functional.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                additiveMask,
                p: 0,
                is_casual: false)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, queryTokens, heads * headDimension);
        if (store.Contains(prefix + ".to_gate_logits.weight"))
        {
            var gateLogits = Linear(x, prefix + ".to_gate_logits");
            observeTensor?.Invoke(prefix + ".to_gate_logits", gateLogits);
            var gates = gateLogits.sigmoid().mul(2)
                .unsqueeze(-1);
            attended = attended.reshape(batch, queryTokens, heads, headDimension)
                .mul(gates)
                .reshape(batch, queryTokens, heads * headDimension);
        }
        observeTensor?.Invoke(prefix + ".attended", attended);
        var output = Linear(attended, prefix + ".to_out.0");
        observeTensor?.Invoke(prefix + ".to_out", output);
        observeTensor?.Invoke(prefix, output);
        return output;
    }

    private Tensor FeedForward(Tensor input, string prefix, Action<string, Tensor>? observeTensor = null)
    {
        var projected = Linear(input, prefix + ".net.0.proj");
        var activated = nn.functional.gelu(projected, TorchSharp.Modules.GELU.Approximate.tanh);
        var output = Linear(activated, prefix + ".net.2");
        observeTensor?.Invoke(prefix, output);
        return output;
    }

    private Tensor Linear(Tensor input, string prefix)
    {
        var weight = Weight(prefix + ".weight");
        // TorchSharp's scalar promotion can lift BF16 modulation intermediates to FP32 even
        // though torch.nn.Linear autocast in the reference executes with the weight dtype.
        var linearInput = input.dtype == weight.dtype ? input : input.to(weight.dtype);
        var bias = store.Contains(prefix + ".bias") ? Weight(prefix + ".bias") : null;
        var result = TorchSharpRuntime.Linear(linearInput, weight, bias);
        if (adapters.TryGetValue(prefix, out var adapter))
        {
            var a = adapter.A.to(input.dtype, device, non_blocking: false);
            var b = adapter.B.to(input.dtype, device, non_blocking: false);
            result = TorchSharpRuntime.Add(
                result,
                linearInput.matmul(a.transpose(0, 1)).matmul(b.transpose(0, 1)).mul(adapter.Scale));
        }
        return result;
    }

    private Tensor Weight(string name) => store.Load(name, device, config.ComputeDType, cacheTensor: cacheWeights);

    private static Tensor Silu(Tensor input) => nn.functional.silu(input);

    private static Tensor ModulatedRmsNorm(Tensor input, Tensor scale, Tensor shift) =>
        TorchSharpRuntime.Add(RmsNorm(input).mul(scale.add(1)), shift);

    private static Tensor RmsNorm(Tensor input, Tensor? weight = null, double epsilon = 1e-6)
        => TorchSharpRuntime.RmsNorm(input, weight, epsilon);

    private Rope PrepareRope(Tensor positions, int dimension, int[] maximumPositions, int heads)
    {
        Tensor resolved;
        if (positions.dim() == 4)
        {
            resolved = TorchSharpRuntime.Add(
                    positions.narrow(-1, 0, 1).squeeze(-1),
                    positions.narrow(-1, 1, 1).squeeze(-1))
                .div(2);
        }
        else if (positions.dim() == 3)
        {
            resolved = positions;
        }
        else
        {
            throw new ArgumentException("RoPE positions must have rank three or four.", nameof(positions));
        }
        if (resolved.shape[1] != maximumPositions.Length)
        {
            throw new ArgumentException("RoPE position axes do not match checkpoint configuration.", nameof(positions));
        }

        var axes = maximumPositions.Length;
        var frequencyCount = dimension / (2 * axes);
        var exponents = linspace(0, 1, frequencyCount, dtype: ScalarType.Float32, device: device);
        var frequencies = exponents.mul(Math.Log(config.RopeTheta)).exp().mul(Math.PI / 2);
        var components = new List<Tensor>(axes);
        for (var axis = 0; axis < axes; axis++)
        {
            components.Add(
                resolved.narrow(1, axis, 1).squeeze(1)
                    .div(maximumPositions[axis])
                    .mul(2)
                    .sub(1)
                    .unsqueeze(-1)
                    .mul(frequencies));
        }
        // Each component is [B,T,F]; stacking on the last axis yields [B,T,F,axis],
        // which is already the reference generate_freqs layout before flattening.
        var phases = stack(components.ToArray(), dim: -1).flatten(2);
        var expected = dimension / 2;
        var padding = checked((int)(expected - phases.shape[2]));
        var cos = phases.cos();
        var sin = phases.sin();
        if (padding > 0)
        {
            cos = cat([ones([cos.shape[0], cos.shape[1], padding], dtype: cos.dtype, device: device), cos], dim: -1);
            sin = cat([zeros([sin.shape[0], sin.shape[1], padding], dtype: sin.dtype, device: device), sin], dim: -1);
        }
        cos = cos.reshape(cos.shape[0], cos.shape[1], heads, -1).transpose(1, 2).to(config.ComputeDType);
        sin = sin.reshape(sin.shape[0], sin.shape[1], heads, -1).transpose(1, 2).to(config.ComputeDType);
        return new Rope(cos, sin, positions, resolved);
    }

    private static Tensor ApplyRope(Tensor input, Rope rope)
    {
        var batch = input.shape[0];
        var tokens = input.shape[1];
        var heads = rope.Cos.shape[1];
        var reshaped = input.reshape(batch, tokens, heads, -1).transpose(1, 2);
        var half = reshaped.shape[3] / 2;
        var first = reshaped.narrow(3, 0, half);
        var second = reshaped.narrow(3, half, half);
        var rotatedFirst = first.mul(rope.Cos).sub(second.mul(rope.Sin));
        var rotatedSecond = TorchSharpRuntime.Add(second.mul(rope.Cos), first.mul(rope.Sin));
        return cat([rotatedFirst, rotatedSecond], dim: 3)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, tokens, heads * half * 2);
    }

    private void ValidateCheckpointSurface()
    {
        var required = new[]
        {
            Root + "patchify_proj.weight",
            Root + "audio_patchify_proj.weight",
            Root + "transformer_blocks.0.attn1.to_q.weight",
            $"{Root}transformer_blocks.{config.LayerCount - 1}.audio_ff.net.2.weight",
            Root + "proj_out.weight",
            Root + "audio_proj_out.weight",
        };
        var missing = required.Where(name => !store.Contains(name)).ToArray();
        if (missing.Length > 0)
        {
            throw new InvalidDataException(
                $"Checkpoint is not a complete LTX audio-video transformer: missing {string.Join(", ", missing)}.");
        }
    }

    private void ValidateAdapters()
    {
        foreach (var (prefix, adapter) in adapters)
        {
            if (!store.Contains(prefix + ".weight"))
            {
                throw new InvalidDataException($"LoRA target '{prefix}' is not a checkpoint linear layer.");
            }
            var weight = store.Describe(prefix + ".weight");
            if (weight.Shape.Count != 2 || adapter.A.dim() != 2 || adapter.B.dim() != 2 ||
                adapter.A.shape[0] != adapter.B.shape[1] || adapter.A.shape[1] != weight.Shape[1] ||
                adapter.B.shape[0] != weight.Shape[0] || adapter.Scale <= 0 || !double.IsFinite(adapter.Scale))
            {
                throw new ArgumentException($"LoRA adapter '{prefix}' does not match checkpoint weight shape.");
            }
        }
    }

    private static void ValidateInput(
        CheckpointTransformerInput? input,
        int contextDimension,
        int positionAxes,
        string parameterName)
    {
        if (input is null)
        {
            return;
        }
        if (input.Latent.dim() != 3 || input.Latent.shape[2] != 128 || input.Context.dim() != 3 ||
            input.Context.shape[2] != contextDimension || input.Timesteps.dim() != 2 || input.Sigma.dim() != 1 ||
            input.Positions.dim() != 4 || input.Positions.shape[1] != positionAxes || input.Positions.shape[3] != 2 ||
            input.Latent.shape[0] != input.Context.shape[0] || input.Latent.shape[0] != input.Timesteps.shape[0] ||
            input.Latent.shape[0] != input.Sigma.shape[0] || input.Latent.shape[0] != input.Positions.shape[0] ||
            input.Latent.shape[1] != input.Timesteps.shape[1] || input.Latent.shape[1] != input.Positions.shape[2])
        {
            throw new ArgumentException("Transformer modality tensors do not match the production checkpoint layout.", parameterName);
        }
        if (input.ContextMask is not null &&
            (input.ContextMask.dim() != 2 || input.ContextMask.shape[0] != input.Context.shape[0] ||
             input.ContextMask.shape[1] != input.Context.shape[1]))
        {
            throw new ArgumentException("Transformer context mask shape does not match context tokens.", parameterName);
        }
    }

    private sealed record TransformerState(
        Tensor X,
        Tensor Context,
        Tensor? ContextMask,
        Tensor Timestep,
        Tensor EmbeddedTimestep,
        Tensor PromptTimestep,
        Rope Rope,
        Rope? CrossRope,
        Tensor? CrossScaleShiftTimestep,
        Tensor? CrossGateTimestep,
        Tensor SourceTimesteps,
        Tensor SourceSigma,
        bool Enabled)
    { }

    private sealed record Rope(Tensor Cos, Tensor Sin, Tensor SourcePositions, Tensor ResolvedPositions);

    private sealed record CrossAda(Tensor Scale, Tensor Shift, Tensor Gate);

    private sealed record TransformerConfig(
        string ModelVersion,
        int LayerCount,
        int VideoDimension,
        int AudioDimension,
        int VideoHeads,
        int AudioHeads,
        int[] VideoMaximumPositions,
        int[] AudioMaximumPositions,
        int ConnectorHeads,
        int ConnectorLayers,
        int ConnectorRegisters,
        int ConnectorMaximumPosition,
        double TimestepScale,
        double AudioVideoTimestepScale,
        double RopeTheta,
        ScalarType ComputeDType)
    {
        public static TransformerConfig Parse(IReadOnlyDictionary<string, string> metadata)
        {
            if (!metadata.TryGetValue("config", out var rawConfig))
            {
                throw new InvalidDataException("Production transformer checkpoint is missing config metadata.");
            }
            using var document = JsonDocument.Parse(rawConfig);
            var root = document.RootElement.GetProperty("transformer");
            var videoHeads = root.GetProperty("num_attention_heads").GetInt32();
            var videoHeadDimension = root.GetProperty("attention_head_dim").GetInt32();
            var audioHeads = root.GetProperty("audio_num_attention_heads").GetInt32();
            var audioHeadDimension = root.GetProperty("audio_attention_head_dim").GetInt32();
            return new TransformerConfig(
                metadata.GetValueOrDefault("model_version") ?? "unknown",
                root.GetProperty("num_layers").GetInt32(),
                checked(videoHeads * videoHeadDimension),
                checked(audioHeads * audioHeadDimension),
                videoHeads,
                audioHeads,
                root.GetProperty("positional_embedding_max_pos").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                root.GetProperty("audio_positional_embedding_max_pos").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                root.GetProperty("connector_num_attention_heads").GetInt32(),
                root.GetProperty("connector_num_layers").GetInt32(),
                root.GetProperty("connector_num_learnable_registers").GetInt32(),
                root.GetProperty("connector_positional_embedding_max_pos")[0].GetInt32(),
                root.GetProperty("timestep_scale_multiplier").GetDouble(),
                root.GetProperty("av_ca_timestep_scale_multiplier").GetDouble(),
                root.GetProperty("positional_embedding_theta").GetDouble(),
                ScalarType.BFloat16);
        }
    }
}
