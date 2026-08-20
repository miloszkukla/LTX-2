using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public static class VaeExecution
{
    public static Tensor PixelNorm(Tensor input, long dimension = 1, double epsilon = 1e-8)
    {
        ArgumentNullException.ThrowIfNull(input);
        using var scope = NewDisposeScope();
        var rootMeanSquare = input.pow(2).mean([dimension], keepdim: true).add(epsilon).sqrt();
        return input.div(rootMeanSquare).MoveToOuterDisposeScope();
    }

    public static Tensor Patchify(Tensor input, int spatialPatchSize, int temporalPatchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(input);
        ValidatePatchSizes(spatialPatchSize, temporalPatchSize);
        if (spatialPatchSize == 1 && temporalPatchSize == 1)
        {
            return input.alias();
        }

        using var scope = NewDisposeScope();
        Tensor output;
        if (input.dim() == 4)
        {
            var (batch, channels, height, width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
            ValidateDivisible(height, spatialPatchSize, nameof(spatialPatchSize));
            ValidateDivisible(width, spatialPatchSize, nameof(spatialPatchSize));
            output = input.reshape(
                    batch,
                    channels,
                    height / spatialPatchSize,
                    spatialPatchSize,
                    width / spatialPatchSize,
                    spatialPatchSize)
                .permute(0, 1, 5, 3, 2, 4)
                .contiguous()
                .reshape(
                    batch,
                    channels * spatialPatchSize * spatialPatchSize,
                    height / spatialPatchSize,
                    width / spatialPatchSize);
        }
        else if (input.dim() == 5)
        {
            var (batch, channels, frames, height, width) =
                (input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]);
            ValidateDivisible(frames, temporalPatchSize, nameof(temporalPatchSize));
            ValidateDivisible(height, spatialPatchSize, nameof(spatialPatchSize));
            ValidateDivisible(width, spatialPatchSize, nameof(spatialPatchSize));
            output = input.reshape(
                    batch,
                    channels,
                    frames / temporalPatchSize,
                    temporalPatchSize,
                    height / spatialPatchSize,
                    spatialPatchSize,
                    width / spatialPatchSize,
                    spatialPatchSize)
                .permute(0, 1, 3, 7, 5, 2, 4, 6)
                .contiguous()
                .reshape(
                    batch,
                    channels * temporalPatchSize * spatialPatchSize * spatialPatchSize,
                    frames / temporalPatchSize,
                    height / spatialPatchSize,
                    width / spatialPatchSize);
        }
        else
        {
            throw new ArgumentException("VAE patchification requires a four- or five-dimensional tensor.", nameof(input));
        }

        return output.MoveToOuterDisposeScope();
    }

    public static Tensor Unpatchify(Tensor input, int spatialPatchSize, int temporalPatchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(input);
        ValidatePatchSizes(spatialPatchSize, temporalPatchSize);
        if (spatialPatchSize == 1 && temporalPatchSize == 1)
        {
            return input.alias();
        }

        using var scope = NewDisposeScope();
        var divisor = temporalPatchSize * spatialPatchSize * spatialPatchSize;
        if (input.shape[1] % divisor != 0)
        {
            throw new ArgumentException("Input channels are not divisible by the patch volume.", nameof(input));
        }

        Tensor output;
        if (input.dim() == 4)
        {
            var (batch, channels, height, width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
            var baseChannels = channels / (spatialPatchSize * spatialPatchSize);
            output = input.reshape(
                    batch,
                    baseChannels,
                    spatialPatchSize,
                    spatialPatchSize,
                    height,
                    width)
                .permute(0, 1, 4, 3, 5, 2)
                .contiguous()
                .reshape(batch, baseChannels, height * spatialPatchSize, width * spatialPatchSize);
        }
        else if (input.dim() == 5)
        {
            var (batch, channels, frames, height, width) =
                (input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]);
            var baseChannels = channels / divisor;
            output = input.reshape(
                    batch,
                    baseChannels,
                    temporalPatchSize,
                    spatialPatchSize,
                    spatialPatchSize,
                    frames,
                    height,
                    width)
                .permute(0, 1, 5, 2, 6, 4, 7, 3)
                .contiguous()
                .reshape(
                    batch,
                    baseChannels,
                    frames * temporalPatchSize,
                    height * spatialPatchSize,
                    width * spatialPatchSize);
        }
        else
        {
            throw new ArgumentException("VAE unpatchification requires a four- or five-dimensional tensor.", nameof(input));
        }

        return output.MoveToOuterDisposeScope();
    }

    public static Tensor NormalizeChannels(
        Tensor input,
        Tensor means,
        Tensor standardDeviations,
        long dimension = 1)
    {
        return ApplyChannelStatistics(input, means, standardDeviations, dimension, normalize: true);
    }

    public static Tensor DenormalizeChannels(
        Tensor input,
        Tensor means,
        Tensor standardDeviations,
        long dimension = 1)
    {
        return ApplyChannelStatistics(input, means, standardDeviations, dimension, normalize: false);
    }

    private static Tensor ApplyChannelStatistics(
        Tensor input,
        Tensor means,
        Tensor standardDeviations,
        long dimension,
        bool normalize)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(means);
        ArgumentNullException.ThrowIfNull(standardDeviations);
        var rank = input.dim();
        var resolvedDimension = dimension < 0 ? rank + dimension : dimension;
        if (resolvedDimension < 0 || resolvedDimension >= rank || means.dim() != 1 || standardDeviations.dim() != 1 ||
            means.shape[0] != input.shape[resolvedDimension] || standardDeviations.shape[0] != input.shape[resolvedDimension])
        {
            throw new ArgumentException("Channel statistics do not match the selected input dimension.", nameof(means));
        }

        using var scope = NewDisposeScope();
        var broadcastShape = Enumerable.Repeat(1L, checked((int)rank)).ToArray();
        broadcastShape[resolvedDimension] = input.shape[resolvedDimension];
        var broadcastMeans = means.to(input).reshape(broadcastShape);
        var broadcastStandardDeviations = standardDeviations.to(input).reshape(broadcastShape);
        var output = normalize
            ? input.sub(broadcastMeans).div(broadcastStandardDeviations)
            : input.mul(broadcastStandardDeviations).add(broadcastMeans);
        return output.MoveToOuterDisposeScope();
    }

    private static void ValidatePatchSizes(int spatialPatchSize, int temporalPatchSize)
    {
        if (spatialPatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(spatialPatchSize));
        }
        if (temporalPatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temporalPatchSize));
        }
    }

    private static void ValidateDivisible(long value, int divisor, string parameterName)
    {
        if (value % divisor != 0)
        {
            throw new ArgumentException($"Dimension {value} is not divisible by {divisor}.", parameterName);
        }
    }
}
