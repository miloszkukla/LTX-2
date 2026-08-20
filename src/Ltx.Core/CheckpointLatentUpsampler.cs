using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

/// <summary>Production LTX learned x2 spatial latent upsampler.</summary>
public sealed class CheckpointLatentUpsampler : IDisposable
{
    private readonly CheckpointTensorStore checkpointStore;
    private readonly CheckpointTensorStore upsamplerStore;
    private readonly Device device;
    private readonly UpsamplerConfig config;
    private bool disposed;

    public CheckpointLatentUpsampler(
        CheckpointTensorStore checkpointStore,
        string upsamplerPath,
        Device device)
    {
        ArgumentNullException.ThrowIfNull(checkpointStore);
        ArgumentException.ThrowIfNullOrWhiteSpace(upsamplerPath);
        ArgumentNullException.ThrowIfNull(device);
        this.checkpointStore = checkpointStore;
        this.device = device;
        upsamplerStore = new CheckpointTensorStore([upsamplerPath]);
        config = UpsamplerConfig.Parse(upsamplerStore.Metadata);
        if (config.Dimensions != 3 || !config.SpatialUpsample || config.TemporalUpsample ||
            config.SpatialScale != 2 || config.RationalResampler)
        {
            throw new NotSupportedException(
                "The C# production path supports the pinned 3D learned x2 spatial upsampler.");
        }
        if (!checkpointStore.Contains("vae.per_channel_statistics.mean-of-means") ||
            !checkpointStore.Contains("vae.per_channel_statistics.std-of-means") ||
            !upsamplerStore.Contains("initial_conv.weight") ||
            !upsamplerStore.Contains("upsampler.0.weight") ||
            !upsamplerStore.Contains("final_conv.weight"))
        {
            throw new InvalidDataException("Checkpoint/upscaler tensor surface is incomplete.");
        }
    }

    public Tensor Upsample(Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (latent.dim() != 5 || latent.shape[1] != config.InputChannels)
        {
            throw new ArgumentException(
                $"Latent must have shape (batch, {config.InputChannels}, frames, height, width).",
                nameof(latent));
        }

        using var scope = NewDisposeScope();
        var means = CheckpointWeight("vae.per_channel_statistics.mean-of-means")
            .reshape(1, config.InputChannels, 1, 1, 1);
        var deviations = CheckpointWeight("vae.per_channel_statistics.std-of-means")
            .reshape(1, config.InputChannels, 1, 1, 1);
        var x = TorchSharpRuntime.Add(
            latent.to(ScalarType.BFloat16, device, non_blocking: false).mul(deviations),
            means);
        x = Convolution3D(x, "initial_conv");
        x = nn.functional.group_norm(
            x,
            32,
            UpsamplerWeight("initial_norm.weight"),
            UpsamplerWeight("initial_norm.bias"),
            1e-5);
        x = nn.functional.silu(x);

        for (var block = 0; block < config.BlocksPerStage; block++)
        {
            x = Residual(x, $"res_blocks.{block}");
        }

        var batch = x.shape[0];
        var frames = x.shape[2];
        var height = x.shape[3];
        var width = x.shape[4];
        x = x.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(batch * frames, config.MiddleChannels, height, width);
        x = Convolution2D(x, "upsampler.0");
        x = PixelShuffle2D(x, 2);
        x = x.reshape(batch, frames, config.MiddleChannels, height * 2, width * 2)
            .permute(0, 2, 1, 3, 4)
            .contiguous();

        for (var block = 0; block < config.BlocksPerStage; block++)
        {
            x = Residual(x, $"post_upsample_res_blocks.{block}");
        }
        x = Convolution3D(x, "final_conv");
        x = x.sub(means).div(deviations);
        return x.to(latent.dtype).MoveToOuterDisposeScope();
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        upsamplerStore.Dispose();
    }

    private Tensor Residual(Tensor input, string prefix)
    {
        var x = Convolution3D(input, prefix + ".conv1");
        x = nn.functional.group_norm(
            x,
            32,
            UpsamplerWeight(prefix + ".norm1.weight"),
            UpsamplerWeight(prefix + ".norm1.bias"),
            1e-5);
        x = nn.functional.silu(x);
        x = Convolution3D(x, prefix + ".conv2");
        x = nn.functional.group_norm(
            x,
            32,
            UpsamplerWeight(prefix + ".norm2.weight"),
            UpsamplerWeight(prefix + ".norm2.bias"),
            1e-5);
        return nn.functional.silu(TorchSharpRuntime.Add(x, input));
    }

    private Tensor Convolution3D(Tensor input, string prefix) => nn.functional.conv3d(
        input,
        UpsamplerWeight(prefix + ".weight"),
        upsamplerStore.Contains(prefix + ".bias") ? UpsamplerWeight(prefix + ".bias") : null,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1);

    private Tensor Convolution2D(Tensor input, string prefix) => nn.functional.conv2d(
        input,
        UpsamplerWeight(prefix + ".weight"),
        upsamplerStore.Contains(prefix + ".bias") ? UpsamplerWeight(prefix + ".bias") : null,
        [1, 1],
        [1, 1],
        [1, 1],
        1);

    private static Tensor PixelShuffle2D(Tensor input, int factor)
    {
        var factorSquared = checked(factor * factor);
        if (input.shape[1] % factorSquared != 0)
        {
            throw new InvalidDataException("Upsampler channels do not divide by the spatial shuffle factor.");
        }
        var channels = input.shape[1] / factorSquared;
        return input.reshape(input.shape[0], channels, factor, factor, input.shape[2], input.shape[3])
            .permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .reshape(input.shape[0], channels, input.shape[2] * factor, input.shape[3] * factor);
    }

    private Tensor CheckpointWeight(string name) =>
        checkpointStore.Load(name, device, ScalarType.BFloat16, cacheTensor: true);

    private Tensor UpsamplerWeight(string name) =>
        upsamplerStore.Load(name, device, ScalarType.BFloat16, cacheTensor: true);

    private sealed record UpsamplerConfig(
        int InputChannels,
        int MiddleChannels,
        int BlocksPerStage,
        int Dimensions,
        bool SpatialUpsample,
        bool TemporalUpsample,
        double SpatialScale,
        bool RationalResampler)
    {
        public static UpsamplerConfig Parse(IReadOnlyDictionary<string, string> metadata)
        {
            if (!metadata.TryGetValue("config", out var raw))
            {
                throw new InvalidDataException("Latent upsampler is missing config metadata.");
            }
            using var document = JsonDocument.Parse(raw);
            var root = document.RootElement;
            if (root.GetProperty("_class_name").GetString() != "LatentUpsampler")
            {
                throw new InvalidDataException("Checkpoint is not an LTX LatentUpsampler.");
            }
            return new UpsamplerConfig(
                root.GetProperty("in_channels").GetInt32(),
                root.GetProperty("mid_channels").GetInt32(),
                root.GetProperty("num_blocks_per_stage").GetInt32(),
                root.GetProperty("dims").GetInt32(),
                root.GetProperty("spatial_upsample").GetBoolean(),
                root.GetProperty("temporal_upsample").GetBoolean(),
                root.GetProperty("spatial_scale").GetDouble(),
                root.GetProperty("rational_resampler").GetBoolean());
        }
    }
}
