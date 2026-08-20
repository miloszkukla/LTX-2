using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

/// <summary>Production BigVGAN-v2 vocoder and bandwidth-extension runtime loaded from an LTX checkpoint.</summary>
public sealed class CheckpointVocoder
{
    private const string Root = "vocoder.";
    private readonly CheckpointTensorStore store;
    private readonly Device device;
    private readonly Config config;

    public CheckpointVocoder(CheckpointTensorStore store, Device device)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(device);
        this.store = store;
        this.device = device;
        config = Config.Parse(store.Metadata);
        if (!store.Contains(Root + "vocoder.conv_pre.weight") ||
            !store.Contains(Root + "bwe_generator.conv_pre.weight") ||
            !store.Contains(Root + "mel_stft.stft_fn.forward_basis") ||
            !store.Contains(Root + "mel_stft.mel_basis"))
        {
            throw new InvalidDataException("Checkpoint does not contain the complete production vocoder/BWE stack.");
        }
    }

    public int OutputSampleRate => config.OutputSampleRate;

    /// <summary>Convert stereo mel features (B, 2, time, 64) to a clipped 48 kHz waveform (B, 2, samples).</summary>
    public Tensor Decode(Tensor mel)
    {
        ArgumentNullException.ThrowIfNull(mel);
        if (mel.dim() != 4 || mel.shape[1] != 2 || mel.shape[3] != config.MelBins)
        {
            throw new ArgumentException($"Vocoder input must have shape (batch, 2, time, {config.MelBins}).", nameof(mel));
        }

        using var scope = NewDisposeScope();
        var outputDType = mel.dtype;
        var x = Generator(mel.to(ScalarType.Float32, device, non_blocking: false), Root + "vocoder.", config.Main, true);
        var lowRateLength = x.shape[2];
        var outputLength = lowRateLength * config.OutputSampleRate / config.InputSampleRate;
        var remainder = lowRateLength % config.HopLength;
        if (remainder != 0)
        {
            x = nn.functional.pad(x, [0, config.HopLength - remainder]);
        }

        var batch = x.shape[0];
        var channels = x.shape[1];
        var flat = x.reshape(batch * channels, x.shape[2]);
        var leftPadding = Math.Max(0, config.FftSize - config.HopLength);
        flat = nn.functional.pad(flat.unsqueeze(1), [leftPadding, 0]);
        var spectrum = nn.functional.conv1d(
            flat,
            Weight(Root + "mel_stft.stft_fn.forward_basis"),
            stride: config.HopLength);
        var frequencies = spectrum.shape[1] / 2;
        var real = spectrum.narrow(1, 0, frequencies);
        var imaginary = spectrum.narrow(1, frequencies, frequencies);
        var magnitude = TorchSharpRuntime.Add(real.pow(2), imaginary.pow(2)).sqrt();
        var melBasis = Weight(Root + "mel_stft.mel_basis");
        var logMel = melBasis.matmul(magnitude).clamp(min: 1e-5).log()
            .reshape(batch, channels, config.MelBins, magnitude.shape[2])
            .transpose(2, 3);

        var residual = Generator(logMel, Root + "bwe_generator.", config.BandwidthExtension, false);
        var skip = HannUpsample(x, config.OutputSampleRate / config.InputSampleRate);
        if (!residual.shape.SequenceEqual(skip.shape))
        {
            throw new InvalidDataException(
                $"Vocoder BWE residual [{string.Join(',', residual.shape)}] and skip " +
                $"[{string.Join(',', skip.shape)}] shapes differ.");
        }
        return TorchSharpRuntime.Add(residual, skip).clamp(-1, 1)
            .narrow(2, 0, outputLength)
            .to(outputDType)
            .MoveToOuterDisposeScope();
    }

    private Tensor Generator(Tensor mel, string prefix, GeneratorConfig generator, bool applyFinalActivation)
    {
        var x = mel.transpose(2, 3).contiguous()
            .reshape(mel.shape[0], 2 * mel.shape[3], mel.shape[2]);
        x = Convolution(x, prefix + "conv_pre", padding: 3);
        for (var stage = 0; stage < generator.UpsampleRates.Length; stage++)
        {
            var stride = generator.UpsampleRates[stage];
            var kernel = generator.UpsampleKernels[stage];
            x = TransposedConvolution(x, $"{prefix}ups.{stage}", stride, (kernel - stride) / 2);
            var outputs = new Tensor[generator.ResidualKernels.Length];
            for (var branch = 0; branch < outputs.Length; branch++)
            {
                outputs[branch] = AmpBlock(
                    x,
                    $"{prefix}resblocks.{stage * outputs.Length + branch}",
                    generator.ResidualKernels[branch],
                    generator.ResidualDilations[branch]);
            }
            x = stack(outputs, 0).mean([0L]);
        }
        x = Activation(x, prefix + "act_post");
        x = Convolution(x, prefix + "conv_post", padding: 3);
        return applyFinalActivation ? x.clamp(-1, 1) : x;
    }

    private Tensor AmpBlock(Tensor input, string prefix, int kernel, int[] dilations)
    {
        var x = input;
        for (var layer = 0; layer < dilations.Length; layer++)
        {
            var dilation = dilations[layer];
            var residual = Activation(x, $"{prefix}.acts1.{layer}");
            residual = Convolution(
                residual,
                $"{prefix}.convs1.{layer}",
                padding: (kernel * dilation - dilation) / 2,
                dilation: dilation);
            residual = Activation(residual, $"{prefix}.acts2.{layer}");
            residual = Convolution(residual, $"{prefix}.convs2.{layer}", padding: (kernel - 1) / 2);
            x = TorchSharpRuntime.Add(x, residual);
        }
        return x;
    }

    private Tensor Activation(Tensor input, string prefix)
    {
        var channels = input.shape[1];
        var upFilter = Weight(prefix + ".upsample.filter").expand(channels, -1, -1);
        var up = nn.functional.pad(input, [5, 5], PaddingModes.Replicate);
        up = nn.functional.conv_transpose1d(up, upFilter, stride: 2, groups: channels).mul(2)
            .narrow(2, 15, input.shape[2] * 2);
        var alpha = Weight(prefix + ".act.alpha").exp().reshape(1, channels, 1);
        var beta = Weight(prefix + ".act.beta").exp().reshape(1, channels, 1);
        up = TorchSharpRuntime.Add(up, up.mul(alpha).sin().pow(2).div(beta.add(1e-9)));
        var downFilter = Weight(prefix + ".downsample.lowpass.filter").expand(channels, -1, -1);
        up = nn.functional.pad(up, [5, 6], PaddingModes.Replicate);
        return nn.functional.conv1d(up, downFilter, stride: 2, groups: channels);
    }

    private Tensor HannUpsample(Tensor input, int ratio)
    {
        var width = checked((int)Math.Ceiling(6 / 0.99));
        var kernel = 2 * width * ratio + 1;
        var time = arange(0, kernel, dtype: ScalarType.Float32, device: device)
            .div(ratio).sub(width).mul(0.99);
        var clamped = time.clamp(-6, 6);
        var window = clamped.mul(Math.PI / 12).cos().pow(2);
        var piTime = time.mul(Math.PI);
        var sinc = where(time.eq(0), ones_like(time), piTime.sin().div(piTime));
        var filter = sinc.mul(window).mul(0.99 / ratio).reshape(1, 1, kernel)
            .expand(input.shape[1], -1, -1);
        var padded = nn.functional.pad(input, [width, width], PaddingModes.Replicate);
        var upsampled = nn.functional.conv_transpose1d(
            padded,
            filter,
            stride: ratio,
            groups: input.shape[1]).mul(ratio);
        var left = 2 * width * ratio;
        var right = kernel - ratio;
        return upsampled.narrow(2, left, upsampled.shape[2] - left - right);
    }

    private Tensor Convolution(Tensor input, string prefix, int padding, int dilation = 1)
    {
        var weight = Weight(prefix + ".weight");
        var bias = store.Contains(prefix + ".bias") ? Weight(prefix + ".bias") : null;
        return nn.functional.conv1d(input, weight, bias, padding: padding, dilation: dilation);
    }

    private Tensor TransposedConvolution(Tensor input, string prefix, int stride, int padding)
    {
        var weight = Weight(prefix + ".weight");
        var bias = store.Contains(prefix + ".bias") ? Weight(prefix + ".bias") : null;
        return nn.functional.conv_transpose1d(input, weight, bias, stride: stride, padding: padding);
    }

    private Tensor Weight(string name) => store.Load(name, device, ScalarType.Float32, cacheTensor: true);

    private sealed record GeneratorConfig(
        int[] UpsampleRates,
        int[] UpsampleKernels,
        int[] ResidualKernels,
        int[][] ResidualDilations);

    private sealed record Config(
        GeneratorConfig Main,
        GeneratorConfig BandwidthExtension,
        int InputSampleRate,
        int OutputSampleRate,
        int HopLength,
        int FftSize,
        int MelBins)
    {
        public static Config Parse(IReadOnlyDictionary<string, string> metadata)
        {
            if (!metadata.TryGetValue("config", out var raw))
            {
                throw new InvalidDataException("Vocoder checkpoint is missing config metadata.");
            }
            using var document = JsonDocument.Parse(raw);
            var root = document.RootElement.GetProperty("vocoder");
            var main = root.GetProperty("vocoder");
            var bwe = root.GetProperty("bwe");
            static GeneratorConfig Generator(JsonElement element)
            {
                if (element.GetProperty("resblock").GetString() != "AMP1" ||
                    element.GetProperty("activation").GetString() != "snakebeta" ||
                    !element.GetProperty("stereo").GetBoolean())
                {
                    throw new NotSupportedException("Only the production stereo AMP1/SnakeBeta vocoder is supported.");
                }
                return new GeneratorConfig(
                    element.GetProperty("upsample_rates").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    element.GetProperty("upsample_kernel_sizes").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    element.GetProperty("resblock_kernel_sizes").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    element.GetProperty("resblock_dilation_sizes").EnumerateArray()
                        .Select(row => row.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray());
            }
            return new Config(
                Generator(main),
                Generator(bwe),
                bwe.GetProperty("input_sampling_rate").GetInt32(),
                bwe.GetProperty("output_sampling_rate").GetInt32(),
                bwe.GetProperty("hop_length").GetInt32(),
                bwe.GetProperty("n_fft").GetInt32(),
                bwe.GetProperty("num_mels").GetInt32());
        }
    }
}
