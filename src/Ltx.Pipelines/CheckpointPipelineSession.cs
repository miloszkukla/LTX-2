using System.Security.Cryptography;
using System.Text;
using Ltx.Core;
using Ltx.Media;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Pipelines;

public sealed record CheckpointPipelineOutput(RgbVideo? Video, AudioData? Audio, bool TwoStage);

/// <summary>Reusable, single-GPU production-checkpoint inference session for the pipeline registry.</summary>
public sealed class CheckpointPipelineSession : IDisposable
{
    private static readonly float[] DistilledSigmas = [1F, 0.99375F, 0.9875F, 0.98125F, 0.975F, 0.909375F, 0.725F, 0.421875F, 0F];
    private static readonly float[] StageTwoDistilledSigmas = [0.909375F, 0.725F, 0.421875F, 0F];
    private readonly CheckpointTensorStore store;
    private readonly CheckpointTransformerRuntime transformer;
    private readonly CheckpointVideoDecoder videoDecoder;
    private readonly CheckpointAudioDecoder audioDecoder;
    private readonly CheckpointVocoder vocoder;
    private readonly CheckpointLatentUpsampler? upsampler;
    private readonly SafeTensorIndex contexts;
    private bool disposed;

    public CheckpointPipelineSession(
        string checkpointPath,
        string textEmbeddingsPath,
        string? torchSharpLibraryPath = null,
        string? spatialUpsamplerPath = null,
        string offloadMode = "none")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(textEmbeddingsPath);
        if (offloadMode is not ("none" or "disk"))
        {
            throw new ArgumentOutOfRangeException(nameof(offloadMode));
        }
        if (torchSharpLibraryPath is not null)
        {
            TorchSharpRuntime.Initialize(torchSharpLibraryPath);
        }
        InitializeDeviceType(DeviceType.CUDA);
        if (!cuda.is_available() || cuda.device_count() != 1)
        {
            throw new InvalidOperationException("Checkpoint pipelines require exactly one CUDA device.");
        }
        if (spatialUpsamplerPath is null)
        {
            TorchSharpRuntime.UseMathSdpOnly();
        }
        else
        {
            TorchSharpRuntime.UsePythonSdpPriority();
        }
        store = new CheckpointTensorStore([checkpointPath]);
        contexts = SafeTensorIndex.Open(textEmbeddingsPath);
        if (!contexts.Tensors.ContainsKey("video_context") || !contexts.Tensors.ContainsKey("audio_context"))
        {
            throw new InvalidDataException("Text embeddings must contain video_context and audio_context tensors.");
        }
        transformer = new CheckpointTransformerRuntime(store, CUDA, cacheWeights: offloadMode == "none");
        videoDecoder = new CheckpointVideoDecoder(store, CUDA);
        audioDecoder = new CheckpointAudioDecoder(store, CUDA);
        vocoder = new CheckpointVocoder(store, CUDA);
        upsampler = spatialUpsamplerPath is null
            ? null
            : new CheckpointLatentUpsampler(store, spatialUpsamplerPath, CUDA);
    }

    public string ModelVersion => transformer.ModelVersion;

    public CheckpointPipelineOutput Generate(PipelineMode mode, PipelineRequest request)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ValidateGeometry(request);
        ValidatePrompt(request.Prompt);
        using var scope = NewDisposeScope();
        manual_seed(request.Seed);
        cuda.manual_seed_all(request.Seed);
        var videoEnabled = !mode.AudioOnly;
        var audioEnabled = !mode.HdrBatch;
        var twoStage = mode.ModuleName == "ltx_pipelines.ti2vid_two_stages" && upsampler is not null;
        var stageOneHeight = twoStage ? request.Height / 2 : request.Height;
        var stageOneWidth = twoStage ? request.Width / 2 : request.Width;
        var latentFrames = (request.FrameCount - 1) / 8 + 1;
        var latentHeight = stageOneHeight / 32;
        var latentWidth = stageOneWidth / 32;
        var videoTokens = checked(latentFrames * latentHeight * latentWidth);
        var duration = request.FrameCount / request.FramesPerSecond;
        var audioTokens = Math.Max(1, checked((int)Math.Round(duration * 25)));
        var videoLatent = videoEnabled
            ? randn([1, videoTokens, 128], dtype: ScalarType.BFloat16, device: CUDA)
            : null;
        var audioLatent = audioEnabled
            ? randn([1, audioTokens, 128], dtype: ScalarType.BFloat16, device: CUDA)
            : null;
        Dictionary<string, SafeTensor>? latentDiagnostics = request.LatentDiagnosticsPath is not null
            ? new Dictionary<string, SafeTensor>(StringComparer.Ordinal)
            : null;
        using var videoContext = videoEnabled ? contexts.LoadTorchTensor("video_context", CUDA) : null;
        using var audioContext = audioEnabled ? contexts.LoadTorchTensor("audio_context", CUDA) : null;
        using var videoPositions = videoEnabled
            ? VideoPositions(latentFrames, latentHeight, latentWidth, request.FramesPerSecond)
            : null;
        using var audioPositions = audioEnabled ? AudioPositions(audioTokens) : null;

        (videoLatent, audioLatent) = Denoise(
            videoLatent,
            audioLatent,
            videoContext,
            audioContext,
            videoPositions,
            audioPositions,
            videoTokens,
            audioTokens,
            ResolveSigmas(request.InferenceSteps),
            latentDiagnostics,
            "stage1");

        if (latentDiagnostics is not null && videoLatent is not null && audioLatent is not null)
        {
            using var stageOneVideo = videoLatent.reshape(1, latentFrames, latentHeight, latentWidth, 128)
                .permute(0, 4, 1, 2, 3).contiguous();
            using var stageOneAudio = audioLatent.reshape(1, audioTokens, 8, 16)
                .permute(0, 2, 1, 3).contiguous();
            latentDiagnostics["stage1_video"] = SafeTensor.FromTorchTensor(stageOneVideo);
            latentDiagnostics["stage1_audio"] = SafeTensor.FromTorchTensor(stageOneAudio);
        }

        if (twoStage)
        {
            using var lowResolutionLatent = videoLatent!.reshape(
                    1, latentFrames, latentHeight, latentWidth, 128)
                .permute(0, 4, 1, 2, 3)
                .contiguous();
            using var upscaled = upsampler!.Upsample(lowResolutionLatent);
            var stageTwoLatentHeight = request.Height / 32;
            var stageTwoLatentWidth = request.Width / 32;
            var stageTwoVideoTokens = checked(latentFrames * stageTwoLatentHeight * stageTwoLatentWidth);
            using var unnoisedVideo = upscaled.permute(0, 2, 3, 4, 1)
                .contiguous()
                .reshape(1, stageTwoVideoTokens, 128);
            var previousVideo = videoLatent;
            videoLatent = AddNoise(unnoisedVideo, StageTwoDistilledSigmas[0]);
            previousVideo.Dispose();
            var previousAudio = audioLatent!;
            audioLatent = AddNoise(previousAudio, StageTwoDistilledSigmas[0]);
            previousAudio.Dispose();
            using var stageTwoVideoPositions = VideoPositions(
                latentFrames,
                stageTwoLatentHeight,
                stageTwoLatentWidth,
                request.FramesPerSecond);
            (videoLatent, audioLatent) = Denoise(
                videoLatent,
                audioLatent,
                videoContext,
                audioContext,
                stageTwoVideoPositions,
                audioPositions,
                stageTwoVideoTokens,
                audioTokens,
                StageTwoDistilledSigmas,
                latentDiagnostics,
                "stage2");

            if (latentDiagnostics is not null)
            {
                using var stageTwoVideo = videoLatent!.reshape(
                        1, latentFrames, stageTwoLatentHeight, stageTwoLatentWidth, 128)
                    .permute(0, 4, 1, 2, 3).contiguous();
                using var stageTwoAudio = audioLatent!.reshape(1, audioTokens, 8, 16)
                    .permute(0, 2, 1, 3).contiguous();
                latentDiagnostics["stage2_video"] = SafeTensor.FromTorchTensor(stageTwoVideo);
                latentDiagnostics["stage2_audio"] = SafeTensor.FromTorchTensor(stageTwoAudio);
            }
        }

        if (latentDiagnostics is not null)
        {
            new SafeTensorFile(
                latentDiagnostics,
                new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["fixture_revision"] = "m7c-csharp-two-stage-latents-v1",
                    ["prompt_sha256"] = Convert.ToHexStringLower(
                        SHA256.HashData(Encoding.UTF8.GetBytes(request.Prompt))),
                    ["seed"] = request.Seed.ToString(),
                    ["frames"] = request.FrameCount.ToString(),
                    ["frame_rate"] = request.FramesPerSecond.ToString(
                        System.Globalization.CultureInfo.InvariantCulture),
                    ["width"] = request.Width.ToString(),
                    ["height"] = request.Height.ToString(),
                }).Save(request.LatentDiagnosticsPath!);
        }

        RgbVideo? video = null;
        AudioData? audio = null;
        if (videoEnabled)
        {
            var finalLatentHeight = request.Height / 32;
            var finalLatentWidth = request.Width / 32;
            using var latent = videoLatent!.reshape(1, latentFrames, finalLatentHeight, finalLatentWidth, 128)
                .permute(0, 4, 1, 2, 3).contiguous();
            using var decoded = videoDecoder.DecodeTiled(latent);
            video = ToVideo(decoded, request);
        }
        if (audioEnabled)
        {
            using var latent = audioLatent!.reshape(1, audioTokens, 8, 16)
                .permute(0, 2, 1, 3).contiguous();
            using var mel = audioDecoder.Decode(latent);
            using var waveform = vocoder.Decode(mel);
            audio = ToAudio(waveform, vocoder.OutputSampleRate);
        }
        return new CheckpointPipelineOutput(video, audio, twoStage);
    }

    private (Tensor? Video, Tensor? Audio) Denoise(
        Tensor? videoLatent,
        Tensor? audioLatent,
        Tensor? videoContext,
        Tensor? audioContext,
        Tensor? videoPositions,
        Tensor? audioPositions,
        int videoTokens,
        int audioTokens,
        IReadOnlyList<float> sigmas,
        Dictionary<string, SafeTensor>? diagnostics = null,
        string? diagnosticPrefix = null)
    {
        using var scope = NewDisposeScope();
        var videoEnabled = videoLatent is not null;
        var audioEnabled = audioLatent is not null;
        for (var step = 0; step < sigmas.Count - 1; step++)
        {
            using var stepScope = NewDisposeScope();
            var sigma = sigmas[step];
            var next = sigmas[step + 1];
            using var sigmaTensor = tensor([sigma], dtype: ScalarType.Float32, device: CUDA);
            using var videoTimesteps = videoEnabled
                ? full([1, videoTokens], sigma, dtype: ScalarType.Float32, device: CUDA)
                : null;
            using var audioTimesteps = audioEnabled
                ? full([1, audioTokens], sigma, dtype: ScalarType.Float32, device: CUDA)
                : null;
            using var output = transformer.Forward(
                videoEnabled
                    ? new CheckpointTransformerInput(
                        videoLatent!, videoContext!, null, videoTimesteps!, sigmaTensor, videoPositions!)
                    : null,
                audioEnabled
                    ? new CheckpointTransformerInput(
                        audioLatent!, audioContext!, null, audioTimesteps!, sigmaTensor, audioPositions!)
                    : null);
            if (step == 0 && diagnostics is not null && diagnosticPrefix is not null)
            {
                if (videoEnabled)
                {
                    diagnostics[$"{diagnosticPrefix}_step0_video_input"] =
                        SafeTensor.FromTorchTensor(videoLatent!);
                    diagnostics[$"{diagnosticPrefix}_step0_video_velocity"] =
                        SafeTensor.FromTorchTensor(output.Video!);
                }
                if (audioEnabled)
                {
                    diagnostics[$"{diagnosticPrefix}_step0_audio_input"] =
                        SafeTensor.FromTorchTensor(audioLatent!);
                    diagnostics[$"{diagnosticPrefix}_step0_audio_velocity"] =
                        SafeTensor.FromTorchTensor(output.Audio!);
                }
            }
            if (videoEnabled)
            {
                var previous = videoLatent!;
                videoLatent = EulerStep(previous, output.Video!, sigma, next)
                    .MoveToOuterDisposeScope();
                previous.Dispose();
            }
            if (audioEnabled)
            {
                var previous = audioLatent!;
                audioLatent = EulerStep(previous, output.Audio!, sigma, next)
                    .MoveToOuterDisposeScope();
                previous.Dispose();
            }
        }

        return (
            videoLatent?.MoveToOuterDisposeScope(),
            audioLatent?.MoveToOuterDisposeScope());
    }

    private static Tensor EulerStep(Tensor sample, Tensor rawVelocity, float sigma, float nextSigma)
    {
        // Match X0Model + EulerDiffusionStep exactly. The reference first converts the raw
        // velocity to a BF16 denoised sample, then reconstructs a BF16 velocity before taking
        // the FP32 Euler update. The two BF16 round-trips are numerically observable.
        using var sampleFloat = sample.to(ScalarType.Float32);
        using var rawVelocityFloat = rawVelocity.to(ScalarType.Float32);
        using var negativeScaledVelocity = rawVelocityFloat.mul(-sigma);
        using var denoisedFloat = TorchSharpRuntime.Add(sampleFloat, negativeScaledVelocity);
        using var denoised = denoisedFloat.to(sample.dtype);
        using var denoisedAsFloat = denoised.to(ScalarType.Float32);
        using var negativeDenoised = denoisedAsFloat.mul(-1);
        using var difference = TorchSharpRuntime.Add(sampleFloat, negativeDenoised);
        using var effectiveVelocity = difference.mul(1F / sigma).to(sample.dtype);
        using var effectiveVelocityFloat = effectiveVelocity.to(ScalarType.Float32);
        using var scaledStep = effectiveVelocityFloat.mul(nextSigma - sigma);
        return TorchSharpRuntime.Add(sampleFloat, scaledStep).to(sample.dtype);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        transformer.Dispose();
        upsampler?.Dispose();
        store.Dispose();
    }

    private static Tensor AddNoise(Tensor initialLatent, float noiseScale)
    {
        var noise = randn_like(initialLatent);
        return TorchSharpRuntime.Add(
                initialLatent.to(ScalarType.Float32).mul(1 - noiseScale),
                noise.to(ScalarType.Float32).mul(noiseScale))
            .to(ScalarType.BFloat16);
    }

    private void ValidatePrompt(string prompt)
    {
        if (!contexts.Metadata.TryGetValue("prompt_sha256", out var expected)) return;
        var actual = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(prompt)));
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException("The supplied prompt does not match the prepared text embeddings.");
        }
    }

    private static float[] ResolveSigmas(int steps)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        if (steps == DistilledSigmas.Length - 1) return DistilledSigmas;
        return Enumerable.Range(0, steps + 1).Select(index => 1F - index / (float)steps).ToArray();
    }

    private static Tensor VideoPositions(int frames, int height, int width, double fps)
    {
        var values = new float[checked(3 * frames * height * width * 2)];
        var token = 0;
        for (var frame = 0; frame < frames; frame++)
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++, token++)
        {
            Set(0, token, frame == 0 ? 0 : (float)((1 + (frame - 1) * 8) / fps));
            Set(0, token, (float)((1 + frame * 8) / fps), 1);
            Set(1, token, y * 32);
            Set(1, token, (y + 1) * 32, 1);
            Set(2, token, x * 32);
            Set(2, token, (x + 1) * 32, 1);
        }
        return tensor(values, [1, 3, token, 2], dtype: ScalarType.Float32, device: CUDA);

        void Set(int axis, int index, float value, int bound = 0) =>
            values[(axis * frames * height * width + index) * 2 + bound] = value;
    }

    private static Tensor AudioPositions(int tokens)
    {
        var values = new float[checked(tokens * 2)];
        for (var token = 0; token < tokens; token++)
        {
            values[token * 2] = Math.Max(token * 4 + 1 - 4, 0) * 0.01F;
            values[token * 2 + 1] = Math.Max((token + 1) * 4 + 1 - 4, 0) * 0.01F;
        }
        return tensor(values, [1, 1, tokens, 2], dtype: ScalarType.Float32, device: CUDA);
    }

    private static RgbVideo ToVideo(Tensor decoded, PipelineRequest request)
    {
        using var resolved = decoded.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
        var sourceFrames = checked((int)resolved.shape[2]);
        var sourceHeight = checked((int)resolved.shape[3]);
        var sourceWidth = checked((int)resolved.shape[4]);
        var frames = Math.Min(request.FrameCount, sourceFrames);
        var height = Math.Min(request.Height, sourceHeight);
        var width = Math.Min(request.Width, sourceWidth);
        var values = resolved.data<float>().ToArray();
        var pixels = new byte[checked(frames * height * width * 3)];
        for (var frame = 0; frame < frames; frame++)
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        for (var channel = 0; channel < 3; channel++)
        {
            var source = (((channel * sourceFrames + frame) * sourceHeight + y) * sourceWidth + x);
            var target = (((frame * height + y) * width + x) * 3 + channel);
            pixels[target] = (byte)Math.Clamp(MathF.Round((values[source] + 1) * 127.5F), 0, 255);
        }
        return new RgbVideo(pixels, width, height, frames, request.FramesPerSecond);
    }

    private static AudioData ToAudio(Tensor waveform, int sampleRate)
    {
        using var resolved = waveform.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
        var channels = checked((int)resolved.shape[1]);
        var samplesPerChannel = checked((int)resolved.shape[2]);
        var planar = resolved.data<float>().ToArray();
        var interleaved = new float[checked(channels * samplesPerChannel)];
        for (var sample = 0; sample < samplesPerChannel; sample++)
        for (var channel = 0; channel < channels; channel++)
        {
            interleaved[sample * channels + channel] = planar[channel * samplesPerChannel + sample];
        }
        return new AudioData(interleaved, sampleRate, channels);
    }

    private static void ValidateGeometry(PipelineRequest request)
    {
        if (request.Width <= 0 || request.Height <= 0 || request.Width % 32 != 0 || request.Height % 32 != 0 ||
            request.FrameCount <= 0 || (request.FrameCount - 1) % 8 != 0 ||
            !double.IsFinite(request.FramesPerSecond) || request.FramesPerSecond <= 0)
        {
            throw new ArgumentException("Checkpoint pipelines require 8k+1 frames, dimensions divisible by 32, and positive FPS.");
        }
    }
}
