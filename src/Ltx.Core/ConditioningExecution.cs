using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public static class ConditioningExecution
{
    public static Tensor ResolveCrossMask(
        Tensor attentionMask,
        long newTokens,
        long batchSize,
        Device device,
        ScalarType dtype)
    {
        ArgumentNullException.ThrowIfNull(attentionMask);
        if (newTokens <= 0 || batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(newTokens));
        }

        using var scope = NewDisposeScope();
        var mask = attentionMask.to(dtype, device);
        Tensor resolved;
        if (mask.dim() == 0)
        {
            resolved = full([batchSize, newTokens], mask.item<float>(), dtype: dtype, device: device);
        }
        else if (mask.dim() == 1)
        {
            if (mask.shape[0] != newTokens)
            {
                throw new ArgumentException("One-dimensional attention mask length must equal newTokens.", nameof(attentionMask));
            }
            resolved = mask.unsqueeze(0).expand(batchSize, -1);
        }
        else if (mask.dim() == 2)
        {
            if (mask.shape[1] != newTokens || (mask.shape[0] != 1 && mask.shape[0] != batchSize))
            {
                throw new ArgumentException("Two-dimensional attention mask shape is incompatible.", nameof(attentionMask));
            }
            resolved = mask.shape[0] == 1 && batchSize > 1 ? mask.expand(batchSize, -1) : mask;
        }
        else
        {
            throw new ArgumentException("Attention mask must be scalar, one-dimensional, or two-dimensional.", nameof(attentionMask));
        }

        return resolved.contiguous().MoveToOuterDisposeScope();
    }

    public static Tensor ResolveCrossMask(
        double attentionStrength,
        long newTokens,
        long batchSize,
        Device device,
        ScalarType dtype)
    {
        if (newTokens <= 0 || batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(newTokens));
        }
        return full([batchSize, newTokens], attentionStrength, dtype: dtype, device: device);
    }

    public static Tensor BuildAttentionMask(
        Tensor? existingMask,
        long noisyTokens,
        long newTokens,
        long existingTokens,
        Tensor crossMask)
    {
        ArgumentNullException.ThrowIfNull(crossMask);
        if (noisyTokens < 0 || newTokens <= 0 || existingTokens < noisyTokens ||
            crossMask.dim() != 2 || crossMask.shape[1] != newTokens)
        {
            throw new ArgumentException("Conditioning mask dimensions are incompatible.", nameof(crossMask));
        }
        if (existingMask is not null &&
            (existingMask.dim() != 3 || existingMask.shape[0] != crossMask.shape[0] ||
             existingMask.shape[1] != existingTokens || existingMask.shape[2] != existingTokens))
        {
            throw new ArgumentException("Existing attention mask dimensions are incompatible.", nameof(existingMask));
        }

        using var scope = NewDisposeScope();
        var batchSize = crossMask.shape[0];
        var totalTokens = existingTokens + newTokens;
        var mask = zeros(
            [batchSize, totalTokens, totalTokens],
            dtype: crossMask.dtype,
            device: crossMask.device);

        var existingBlock = existingMask is null
            ? ones([batchSize, existingTokens, existingTokens], dtype: crossMask.dtype, device: crossMask.device)
            : existingMask.to(crossMask);
        mask[TensorIndex.Colon, TensorIndex.Slice(0, existingTokens), TensorIndex.Slice(0, existingTokens)] = existingBlock;
        mask[TensorIndex.Colon, TensorIndex.Slice(existingTokens), TensorIndex.Slice(existingTokens)] =
            ones([batchSize, newTokens, newTokens], dtype: crossMask.dtype, device: crossMask.device);
        mask[TensorIndex.Colon, TensorIndex.Slice(0, noisyTokens), TensorIndex.Slice(existingTokens)] =
            crossMask.unsqueeze(1).expand(batchSize, noisyTokens, newTokens);
        mask[TensorIndex.Colon, TensorIndex.Slice(existingTokens), TensorIndex.Slice(0, noisyTokens)] =
            crossMask.unsqueeze(2).expand(batchSize, newTokens, noisyTokens);

        return mask.MoveToOuterDisposeScope();
    }
}
