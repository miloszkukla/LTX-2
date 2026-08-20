using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public static class AudioExecution
{
    public static Tensor Snake(
        Tensor input,
        Tensor alpha,
        bool alphaLogScale = true,
        double epsilon = 1e-9)
    {
        return SnakeBeta(input, alpha, alpha, alphaLogScale, epsilon);
    }

    public static Tensor SnakeBeta(
        Tensor input,
        Tensor alpha,
        Tensor beta,
        bool parameterLogScale = true,
        double epsilon = 1e-9)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(alpha);
        ArgumentNullException.ThrowIfNull(beta);
        if (input.dim() < 2 || alpha.dim() != 1 || beta.dim() != 1 ||
            alpha.shape[0] != input.shape[1] || beta.shape[0] != input.shape[1])
        {
            throw new ArgumentException("Snake parameters must match the input channel dimension.", nameof(alpha));
        }

        using var scope = NewDisposeScope();
        var parameterShape = Enumerable.Repeat(1L, checked((int)input.dim())).ToArray();
        parameterShape[1] = input.shape[1];
        var resolvedAlpha = alpha.to(input).reshape(parameterShape);
        var resolvedBeta = beta.to(input).reshape(parameterShape);
        if (parameterLogScale)
        {
            resolvedAlpha = resolvedAlpha.exp();
            resolvedBeta = resolvedBeta.exp();
        }

        var periodic = input.mul(resolvedAlpha).sin().pow(2).div(resolvedBeta.add(epsilon));
        return input.add(periodic).MoveToOuterDisposeScope();
    }

    public static Tensor PrepareStereoMel(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.dim() != 4 || input.shape[1] != 2)
        {
            throw new ArgumentException("Stereo mel input must have shape (batch, 2, time, mel_bins).", nameof(input));
        }

        using var scope = NewDisposeScope();
        var transposed = input.transpose(2, 3).contiguous();
        var output = transposed.reshape(
            transposed.shape[0],
            transposed.shape[1] * transposed.shape[2],
            transposed.shape[3]);
        return output.MoveToOuterDisposeScope();
    }

    public static Tensor ApplyVocoderFinalActivation(Tensor input, bool useTanh)
    {
        ArgumentNullException.ThrowIfNull(input);
        using var scope = NewDisposeScope();
        var output = useTanh ? input.tanh() : input.clamp(-1, 1);
        return output.MoveToOuterDisposeScope();
    }
}
