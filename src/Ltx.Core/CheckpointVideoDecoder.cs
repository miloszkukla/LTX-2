using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

/// <summary>Production convolutional LTX video-VAE decoder backed by checkpoint tensors.</summary>
public sealed class CheckpointVideoDecoder
{
    private readonly CheckpointTensorStore store;
    private readonly Device device;
    private readonly DecoderConfig config;

    public CheckpointVideoDecoder(CheckpointTensorStore store, Device device)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(device);
        this.store = store;
        this.device = device;
        config = DecoderConfig.Parse(store.Metadata);
        if (config.ClassName != "CausalVideoAutoencoder")
        {
            throw new NotSupportedException(
                $"Checkpoint video decoder '{config.ClassName}' is diffusion-based; use its dedicated runtime.");
        }
        if (!store.Contains("vae.decoder.conv_in.conv.weight") || !store.Contains("vae.decoder.conv_out.conv.weight"))
        {
            throw new InvalidDataException("Checkpoint does not contain a complete convolutional video decoder.");
        }
    }

    public Tensor Decode(Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.dim() != 5 || latent.shape[1] != config.LatentChannels)
        {
            throw new ArgumentException(
                $"Video latent must have shape (batch, {config.LatentChannels}, frames, height, width).",
                nameof(latent));
        }

        using var scope = NewDisposeScope();
        var outputDType = latent.dtype;
        var x = latent.to(ScalarType.BFloat16, device, non_blocking: false);
        var means = Weight("vae.per_channel_statistics.mean-of-means")
            .reshape(1, config.LatentChannels, 1, 1, 1);
        var standardDeviations = Weight("vae.per_channel_statistics.std-of-means")
            .reshape(1, config.LatentChannels, 1, 1, 1);
        x = x.mul(standardDeviations).add(means);
        x = Convolution(x, "vae.decoder.conv_in.conv");

        for (var blockIndex = 0; blockIndex < config.Blocks.Count; blockIndex++)
        {
            using var blockScope = NewDisposeScope();
            var previous = x;
            var block = config.Blocks[blockIndex];
            if (block.Name == "res_x")
            {
                for (var layer = 0; layer < block.Layers; layer++)
                {
                    var residualPrefix = $"vae.decoder.up_blocks.{blockIndex}.res_blocks.{layer}";
                    x = Residual(x, residualPrefix);
                }
            }
            else
            {
                var stride = block.Name switch
                {
                    "compress_time" => new[] { 2, 1, 1 },
                    "compress_space" => new[] { 1, 2, 2 },
                    "compress_all" => new[] { 2, 2, 2 },
                    _ => throw new NotSupportedException($"Unsupported convolutional decoder block '{block.Name}'."),
                };
                x = DepthToSpace(
                    Convolution(x, $"vae.decoder.up_blocks.{blockIndex}.conv.conv"),
                    stride[0],
                    stride[1],
                    stride[2]);
            }
            x = x.MoveToOuterDisposeScope();
            previous.Dispose();
        }

        x = VaeExecution.PixelNorm(x);
        x = Silu(x);
        x = Convolution(x, "vae.decoder.conv_out.conv");
        x = VaeExecution.Unpatchify(x, config.PatchSize);
        return x.to(outputDType).MoveToOuterDisposeScope();
    }

    private Tensor Residual(Tensor input, string prefix)
    {
        var x = Silu(VaeExecution.PixelNorm(input));
        x = Convolution(x, prefix + ".conv1.conv");
        x = Silu(VaeExecution.PixelNorm(x));
        x = Convolution(x, prefix + ".conv2.conv");
        return input.add(x);
    }

    private Tensor Convolution(Tensor input, string prefix)
    {
        var weight = Weight(prefix + ".weight");
        var bias = store.Contains(prefix + ".bias") ? Weight(prefix + ".bias") : null;
        var temporalPadding = checked((int)(weight.shape[2] - 1));
        Tensor padded;
        if (config.Causal)
        {
            var first = input.narrow(2, 0, 1).repeat([1, 1, temporalPadding, 1, 1]);
            padded = cat([first, input], dim: 2);
        }
        else
        {
            var side = temporalPadding / 2;
            var first = input.narrow(2, 0, 1).repeat([1, 1, side, 1, 1]);
            var last = input.narrow(2, input.shape[2] - 1, 1).repeat([1, 1, side, 1, 1]);
            padded = cat([first, input, last], dim: 2);
        }

        if (config.ReflectSpatialPadding)
        {
            padded = nn.functional.pad(padded, [1, 1, 1, 1, 0, 0], PaddingModes.Reflect);
            return nn.functional.conv3d(padded, weight, bias, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        }
        return nn.functional.conv3d(padded, weight, bias, [1, 1, 1], [0, 1, 1], [1, 1, 1], 1);
    }

    private static Tensor DepthToSpace(Tensor input, int temporal, int height, int width)
    {
        var divisor = checked(temporal * height * width);
        if (input.shape[1] % divisor != 0)
        {
            throw new InvalidDataException("Video decoder upsample channels do not divide by its stride volume.");
        }
        var channels = input.shape[1] / divisor;
        var output = input.reshape(
                input.shape[0], channels, temporal, height, width,
                input.shape[2], input.shape[3], input.shape[4])
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .contiguous()
            .reshape(
                input.shape[0], channels,
                input.shape[2] * temporal,
                input.shape[3] * height,
                input.shape[4] * width);
        return temporal == 2 ? output.narrow(2, 1, output.shape[2] - 1) : output;
    }

    private Tensor Weight(string name) => store.Load(name, device, cacheTensor: true);

    private static Tensor Silu(Tensor input) => input.mul(input.sigmoid());

    private sealed record DecoderBlock(string Name, int Layers);

    private sealed record DecoderConfig(
        string ClassName,
        int LatentChannels,
        int PatchSize,
        bool Causal,
        bool ReflectSpatialPadding,
        IReadOnlyList<DecoderBlock> Blocks)
    {
        public static DecoderConfig Parse(IReadOnlyDictionary<string, string> metadata)
        {
            if (!metadata.TryGetValue("config", out var raw))
            {
                throw new InvalidDataException("Video VAE checkpoint is missing config metadata.");
            }
            using var document = JsonDocument.Parse(raw);
            var vae = document.RootElement.GetProperty("vae");
            var blocks = new List<DecoderBlock>();
            foreach (var block in vae.GetProperty("decoder_blocks").EnumerateArray().Reverse())
            {
                var name = block[0].GetString()!;
                var values = block[1];
                var layers = values.ValueKind == JsonValueKind.Number
                    ? values.GetInt32()
                    : values.TryGetProperty("num_layers", out var count) ? count.GetInt32() : 0;
                blocks.Add(new DecoderBlock(name, layers));
            }
            var padding = vae.TryGetProperty("spatial_padding_mode", out var paddingMode)
                ? paddingMode.GetString()
                : "reflect";
            return new DecoderConfig(
                vae.GetProperty("_class_name").GetString()!,
                vae.GetProperty("latent_channels").GetInt32(),
                vae.GetProperty("patch_size").GetInt32(),
                vae.GetProperty("causal_decoder").GetBoolean(),
                string.Equals(padding, "reflect", StringComparison.Ordinal),
                blocks);
        }
    }
}
