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
        x = TorchSharpRuntime.Add(x.mul(standardDeviations), means);
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

    /// <summary>
    /// Decodes with the same automatic Conv-VAE tiling geometry used by the Python pipelines:
    /// an aspect-coupled 768 px long side with 64 px spatial overlap and 80 frames with
    /// 24 frames of temporal overlap. Tiles are blended with complementary trapezoidal masks.
    /// </summary>
    public Tensor DecodeTiled(Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.dim() != 5 || latent.shape[1] != config.LatentChannels)
        {
            throw new ArgumentException(
                $"Video latent must have shape (batch, {config.LatentChannels}, frames, height, width).",
                nameof(latent));
        }

        using var scope = NewDisposeScope();
        var temporalScale = config.TemporalScale;
        var heightScale = config.HeightScale;
        var widthScale = config.WidthScale;
        var outputFrames = checked(1 + (latent.shape[2] - 1) * temporalScale);
        var outputHeight = checked(latent.shape[3] * heightScale);
        var outputWidth = checked(latent.shape[4] * widthScale);
        var temporalTiles = SplitBySize(
            checked((int)latent.shape[2]),
            Math.Max(2, 80 / temporalScale),
            24 / temporalScale,
            causal: true);
        var longSide = Math.Max(outputHeight, outputWidth);
        var heightTile = ResolveSpatialTile(outputHeight, longSide, heightScale);
        var widthTile = ResolveSpatialTile(outputWidth, longSide, widthScale);
        var heightTiles = SplitBySize(checked((int)latent.shape[3]), heightTile, 64 / heightScale);
        var widthTiles = SplitBySize(checked((int)latent.shape[4]), widthTile, 64 / widthScale);
        var output = zeros(
            [latent.shape[0], 3, outputFrames, outputHeight, outputWidth],
            dtype: latent.dtype,
            device: device);

        foreach (var temporal in temporalTiles)
        foreach (var height in heightTiles)
        foreach (var width in widthTiles)
        {
            using var tileScope = NewDisposeScope();
            using var latentTile = latent
                .narrow(2, temporal.Start, temporal.Length)
                .narrow(3, height.Start, height.Length)
                .narrow(4, width.Start, width.Length);
            using var decodedTile = Decode(latentTile);
            var temporalOutput = MapTemporal(temporal, temporalScale);
            var heightOutput = MapSpatial(height, heightScale);
            var widthOutput = MapSpatial(width, widthScale);
            if (decodedTile.shape[2] != temporalOutput.Length ||
                decodedTile.shape[3] != heightOutput.Length ||
                decodedTile.shape[4] != widthOutput.Length)
            {
                throw new InvalidDataException("A decoded Conv-VAE tile does not match its mapped output region.");
            }

            using var temporalMask = TrapezoidalMask(
                temporalOutput.Length,
                temporalOutput.LeftRamp,
                temporalOutput.RightRamp,
                leftStartsFromZero: true).reshape(1, 1, temporalOutput.Length, 1, 1);
            using var heightMask = TrapezoidalMask(
                heightOutput.Length,
                heightOutput.LeftRamp,
                heightOutput.RightRamp).reshape(1, 1, 1, heightOutput.Length, 1);
            using var widthMask = TrapezoidalMask(
                widthOutput.Length,
                widthOutput.LeftRamp,
                widthOutput.RightRamp).reshape(1, 1, 1, 1, widthOutput.Length);
            using var weighted = decodedTile
                .mul(temporalMask)
                .mul(heightMask)
                .mul(widthMask);
            using var region = output
                .narrow(2, temporalOutput.Start, temporalOutput.Length)
                .narrow(3, heightOutput.Start, heightOutput.Length)
                .narrow(4, widthOutput.Start, widthOutput.Length);
            using var accumulated = TorchSharpRuntime.Add(region, weighted);
            region.copy_(accumulated);
        }

        return output.MoveToOuterDisposeScope();
    }

    private Tensor Residual(Tensor input, string prefix)
    {
        var x = Silu(VaeExecution.PixelNorm(input));
        x = Convolution(x, prefix + ".conv1.conv");
        x = Silu(VaeExecution.PixelNorm(x));
        x = Convolution(x, prefix + ".conv2.conv");
        return TorchSharpRuntime.Add(input, x);
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

    private static int ResolveSpatialTile(long axis, long longSide, int scale)
    {
        var axisLatent = axis / scale;
        var longLatent = longSide / scale;
        var requestedLatent = 768 / scale;
        var overlapLatent = 64 / scale;
        var rounded = checked((int)Math.Round(
            requestedLatent * axisLatent / (double)longLatent,
            MidpointRounding.ToEven));
        return Math.Max(Math.Max(2, overlapLatent + 1), rounded);
    }

    private static IReadOnlyList<TileInterval> SplitBySize(
        int length,
        int size,
        int overlap,
        bool causal = false)
    {
        if (length <= size)
        {
            return [new TileInterval(0, length, 0, 0)];
        }
        if (size <= 0 || overlap < 0 || overlap >= size)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Tile size and overlap are invalid.");
        }
        var stride = size - overlap;
        var amount = (length + size - 2 * overlap - 1) / stride;
        var intervals = new List<TileInterval>(amount)
        {
            new(0, size, 0, overlap),
        };
        for (var index = 1; index < amount - 1; index++)
        {
            intervals.Add(new(index * stride, index * stride + size, overlap, overlap));
        }
        intervals.Add(new((amount - 1) * stride, length, overlap, 0));
        if (causal)
        {
            for (var index = 1; index < intervals.Count; index++)
            {
                var interval = intervals[index];
                intervals[index] = interval with
                {
                    Start = interval.Start - 1,
                    LeftRamp = interval.LeftRamp + 1,
                };
            }
        }
        return intervals;
    }

    private static TileInterval MapTemporal(TileInterval interval, int scale) => new(
        interval.Start * scale,
        1 + (interval.End - 1) * scale,
        interval.LeftRamp == 0 ? 0 : 1 + (interval.LeftRamp - 1) * scale,
        interval.RightRamp * scale);

    private static TileInterval MapSpatial(TileInterval interval, int scale) => new(
        interval.Start * scale,
        interval.End * scale,
        interval.LeftRamp * scale,
        interval.RightRamp * scale);

    private Tensor TrapezoidalMask(int length, int leftRamp, int rightRamp, bool leftStartsFromZero = false)
    {
        var values = Enumerable.Repeat(1F, length).ToArray();
        if (leftRamp > 0)
        {
            var denominator = leftStartsFromZero ? leftRamp : leftRamp + 1;
            for (var index = 0; index < leftRamp; index++)
            {
                values[index] *= (index + (leftStartsFromZero ? 0 : 1)) / (float)denominator;
            }
        }
        if (rightRamp > 0)
        {
            var denominator = rightRamp + 1F;
            for (var index = 0; index < rightRamp; index++)
            {
                values[length - rightRamp + index] *= (rightRamp - index) / denominator;
            }
        }
        return tensor(values, dtype: ScalarType.Float32, device: device);
    }

    private sealed record DecoderBlock(string Name, int Layers);

    private sealed record TileInterval(int Start, int End, int LeftRamp, int RightRamp)
    {
        public int Length => End - Start;
    }

    private sealed record DecoderConfig(
        string ClassName,
        int LatentChannels,
        int PatchSize,
        bool Causal,
        bool ReflectSpatialPadding,
        IReadOnlyList<DecoderBlock> Blocks)
    {
        public int TemporalScale => Blocks.Aggregate(
            1,
            (scale, block) => block.Name is "compress_time" or "compress_all" ? scale * 2 : scale);

        public int HeightScale => PatchSize * Blocks.Aggregate(
            1,
            (scale, block) => block.Name is "compress_space" or "compress_all" ? scale * 2 : scale);

        public int WidthScale => HeightScale;

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
