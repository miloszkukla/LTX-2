using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public static class TransformerExecution
{
    public static Tensor TimestepEmbedding(
        Tensor timesteps,
        int embeddingDimension,
        bool flipSinToCos = false,
        double downscaleFrequencyShift = 1,
        double scale = 1,
        int maximumPeriod = 10_000)
    {
        ArgumentNullException.ThrowIfNull(timesteps);
        if (timesteps.dim() != 1)
        {
            throw new ArgumentException("Timesteps must be a one-dimensional tensor.", nameof(timesteps));
        }
        if (embeddingDimension < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        }

        var halfDimension = embeddingDimension / 2;
        var denominator = halfDimension - downscaleFrequencyShift;
        if (Math.Abs(denominator) < double.Epsilon)
        {
            throw new ArgumentOutOfRangeException(nameof(downscaleFrequencyShift));
        }

        using var scope = NewDisposeScope();
        var exponent = arange(
                0,
                halfDimension,
                dtype: ScalarType.Float32,
                device: timesteps.device)
            .mul(-Math.Log(maximumPeriod))
            .div(denominator);
        var frequencies = exponent.exp();
        var arguments = timesteps.to(ScalarType.Float32).unsqueeze(1)
            .mul(frequencies.unsqueeze(0))
            .mul(scale);
        var embedding = cat([arguments.sin(), arguments.cos()], dim: -1);

        if (flipSinToCos)
        {
            embedding = cat(
                [
                    embedding.narrow(-1, halfDimension, halfDimension),
                    embedding.narrow(-1, 0, halfDimension),
                ],
                dim: -1);
        }

        if ((embeddingDimension & 1) != 0)
        {
            embedding = cat(
                [
                    embedding,
                    zeros([timesteps.shape[0], 1], dtype: embedding.dtype, device: embedding.device),
                ],
                dim: -1);
        }

        return embedding.MoveToOuterDisposeScope();
    }

    public static Tensor FeedForward(
        Tensor input,
        Tensor inputWeight,
        Tensor inputBias,
        Tensor outputWeight,
        Tensor outputBias)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(inputWeight);
        ArgumentNullException.ThrowIfNull(inputBias);
        ArgumentNullException.ThrowIfNull(outputWeight);
        ArgumentNullException.ThrowIfNull(outputBias);

        using var scope = NewDisposeScope();
        var projected = input.matmul(inputWeight.transpose(0, 1)).add(inputBias);
        var tanhArgument = projected.add(projected.pow(3).mul(0.044715))
            .mul(Math.Sqrt(2 / Math.PI));
        var activated = projected.mul(0.5).mul(tanhArgument.tanh().add(1));
        var output = activated.matmul(outputWeight.transpose(0, 1)).add(outputBias);
        return output.MoveToOuterDisposeScope();
    }

    public static Tensor ScaledDotProductAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        int heads,
        Tensor? mask = null)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3)
        {
            throw new ArgumentException("Attention inputs must have shape (batch, tokens, channels).", nameof(query));
        }
        if (heads <= 0 || query.shape[2] % heads != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(heads));
        }
        if (query.shape[0] != key.shape[0] || key.shape[0] != value.shape[0] ||
            key.shape[1] != value.shape[1] || query.shape[2] != key.shape[2] || key.shape[2] != value.shape[2])
        {
            throw new ArgumentException("Attention input shapes are incompatible.", nameof(key));
        }

        using var scope = NewDisposeScope();
        var batch = query.shape[0];
        var queryTokens = query.shape[1];
        var keyTokens = key.shape[1];
        var headDimension = query.shape[2] / heads;
        var q = query.reshape(batch, queryTokens, heads, headDimension).transpose(1, 2);
        var k = key.reshape(batch, keyTokens, heads, headDimension).transpose(1, 2);
        var v = value.reshape(batch, keyTokens, heads, headDimension).transpose(1, 2);
        var scores = q.matmul(k.transpose(-2, -1)).div(Math.Sqrt(headDimension));

        if (mask is not null)
        {
            var expandedMask = mask.dim() switch
            {
                2 => mask.unsqueeze(0).unsqueeze(0),
                3 => mask.unsqueeze(1),
                4 => mask,
                _ => throw new ArgumentException("Attention mask must have two, three, or four dimensions.", nameof(mask)),
            };
            scores = scores.add(expandedMask.to(scores).mul(-1).add(1).mul(-10_000));
        }

        var attended = scores.softmax(-1).matmul(v)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, queryTokens, heads * headDimension);
        return attended.MoveToOuterDisposeScope();
    }
}
