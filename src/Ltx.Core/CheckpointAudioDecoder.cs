using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

/// <summary>
/// Executes the production LTX audio VAE decoder from checkpoint tensors. The decoder is
/// intentionally functional: checkpoint metadata selects the topology and no Python module is
/// involved at runtime.
/// </summary>
public sealed class CheckpointAudioDecoder
{
    private const string Root = "audio_vae.decoder.";
    private readonly CheckpointTensorStore store;
    private readonly Device device;
    private readonly DecoderConfig config;

    public CheckpointAudioDecoder(CheckpointTensorStore store, Device device)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(device);
        this.store = store;
        this.device = device;
        config = DecoderConfig.Parse(store.Metadata);
        if (!store.Contains(Root + "conv_in.conv.weight") ||
            !store.Contains(Root + "conv_out.conv.weight") ||
            !store.Contains("audio_vae.per_channel_statistics.mean-of-means"))
        {
            throw new InvalidDataException("Checkpoint does not contain a complete production audio VAE decoder.");
        }
    }

    public int SampleRate => config.SampleRate;

    public int MelBins => config.MelBins;

    /// <summary>Decode (B, 8, latent-time, 16) latents to (B, 2, mel-time, 64).</summary>
    public Tensor Decode(Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.dim() != 4 || latent.shape[1] != config.LatentChannels || latent.shape[3] != 16)
        {
            throw new ArgumentException(
                $"Audio latent must have shape (batch, {config.LatentChannels}, frames, 16).",
                nameof(latent));
        }

        using var scope = NewDisposeScope();
        var outputDType = latent.dtype;
        var batch = latent.shape[0];
        var frames = latent.shape[2];
        var x = latent.to(ScalarType.BFloat16, device, non_blocking: false)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, frames, config.LatentChannels * 16);
        var means = Weight("audio_vae.per_channel_statistics.mean-of-means").reshape(1, 1, -1);
        var standardDeviations = Weight("audio_vae.per_channel_statistics.std-of-means").reshape(1, 1, -1);
        x = TorchSharpRuntime.Add(x.mul(standardDeviations), means)
            .reshape(batch, frames, config.LatentChannels, 16)
            .permute(0, 2, 1, 3)
            .contiguous();

        x = Convolution(x, Root + "conv_in.conv");
        x = Residual(x, Root + "mid.block_1");
        x = Residual(x, Root + "mid.block_2");

        for (var level = config.ChannelMultipliers.Length - 1; level >= 0; level--)
        {
            for (var block = 0; block < config.ResidualBlocks + 1; block++)
            {
                x = Residual(x, $"{Root}up.{level}.block.{block}");
            }
            if (level != 0)
            {
                x = NearestNeighbor2x(x);
                x = Convolution(x, $"{Root}up.{level}.upsample.conv.conv");
                // Causal-height upsampling duplicates the first frame twice. The source removes
                // the first result so decoded length is 1 + 2*n at every level.
                x = x.narrow(2, 1, x.shape[2] - 1);
            }
        }

        x = PixelNorm(x);
        x = Silu(x);
        x = Convolution(x, Root + "conv_out.conv");
        var targetFrames = Math.Max(checked(frames * 4 - 3), 1);
        x = x.narrow(1, 0, Math.Min(config.OutputChannels, checked((int)x.shape[1])))
            .narrow(2, 0, Math.Min(targetFrames, checked((int)x.shape[2])))
            .narrow(3, 0, Math.Min(config.MelBins, checked((int)x.shape[3])));
        return x.to(outputDType).MoveToOuterDisposeScope();
    }

    private Tensor Residual(Tensor input, string prefix)
    {
        var x = Convolution(Silu(PixelNorm(input)), prefix + ".conv1.conv");
        x = Convolution(Silu(PixelNorm(x)), prefix + ".conv2.conv");
        if (store.Contains(prefix + ".nin_shortcut.conv.weight"))
        {
            input = Convolution(input, prefix + ".nin_shortcut.conv");
        }
        else if (store.Contains(prefix + ".conv_shortcut.conv.weight"))
        {
            input = Convolution(input, prefix + ".conv_shortcut.conv");
        }
        return TorchSharpRuntime.Add(input, x);
    }

    private Tensor Convolution(Tensor input, string prefix)
    {
        var weight = Weight(prefix + ".weight");
        var bias = store.Contains(prefix + ".bias") ? Weight(prefix + ".bias") : null;
        var heightPadding = checked((int)((weight.shape[2] - 1)));
        var widthPadding = checked((int)((weight.shape[3] - 1)));
        var widthLeft = widthPadding / 2;
        var padded = nn.functional.pad(
            input,
            [widthLeft, widthPadding - widthLeft, heightPadding, 0],
            PaddingModes.Constant);
        return nn.functional.conv2d(padded, weight, bias, [1, 1], [0, 0], [1, 1], 1);
    }

    private static Tensor NearestNeighbor2x(Tensor input) => input
        .unsqueeze(3)
        .unsqueeze(5)
        .expand(input.shape[0], input.shape[1], input.shape[2], 2, input.shape[3], 2)
        .contiguous()
        .reshape(input.shape[0], input.shape[1], input.shape[2] * 2, input.shape[3] * 2);

    private Tensor Weight(string name) => store.Load(name, device, cacheTensor: true);

    private static Tensor PixelNorm(Tensor input) =>
        input.mul(input.pow(2).mean([1L], keepdim: true).add(1e-6).rsqrt());

    private static Tensor Silu(Tensor input) => input.mul(input.sigmoid());

    private sealed record DecoderConfig(
        int LatentChannels,
        int OutputChannels,
        int[] ChannelMultipliers,
        int ResidualBlocks,
        int MelBins,
        int SampleRate)
    {
        public static DecoderConfig Parse(IReadOnlyDictionary<string, string> metadata)
        {
            if (!metadata.TryGetValue("config", out var raw))
            {
                throw new InvalidDataException("Audio VAE checkpoint is missing config metadata.");
            }
            using var document = JsonDocument.Parse(raw);
            var audioVae = document.RootElement.GetProperty("audio_vae");
            var parameters = audioVae.GetProperty("model").GetProperty("params");
            var decoder = parameters.GetProperty("ddconfig");
            var causalAxis = decoder.TryGetProperty("causality_axis", out var axis)
                ? axis.GetString()
                : "height";
            var norm = decoder.TryGetProperty("norm_type", out var normType)
                ? normType.GetString()
                : "group";
            if (causalAxis != "height" || norm != "pixel" ||
                (decoder.TryGetProperty("mid_block_add_attention", out var attention) && attention.GetBoolean()))
            {
                throw new NotSupportedException(
                    "This checkpoint's audio decoder topology is outside the production LTX-2.3/2.5 layout.");
            }
            return new DecoderConfig(
                decoder.GetProperty("z_channels").GetInt32(),
                decoder.GetProperty("out_ch").GetInt32(),
                decoder.GetProperty("ch_mult").EnumerateArray().Select(value => value.GetInt32()).ToArray(),
                decoder.GetProperty("num_res_blocks").GetInt32(),
                decoder.GetProperty("mel_bins").GetInt32(),
                parameters.GetProperty("sampling_rate").GetInt32());
        }
    }
}
