using Ltx.Core;
using Ltx.Media;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Pipelines;

public sealed record CheckpointPipelineOutput(RgbVideo? Video, AudioData? Audio);

/// <summary>Reusable, single-GPU production-checkpoint inference session for the pipeline registry.</summary>
public sealed class CheckpointPipelineSession : IDisposable
{
    private static readonly float[] DistilledSigmas = [1F, 0.99375F, 0.9875F, 0.98125F, 0.975F, 0.909375F, 0.725F, 0.421875F, 0F];
    private readonly CheckpointTensorStore store;
    private readonly CheckpointTransformerRuntime transformer;
    private readonly CheckpointVideoDecoder videoDecoder;
    private readonly CheckpointAudioDecoder audioDecoder;
    private readonly CheckpointVocoder vocoder;
    private readonly SafeTensorIndex contexts;
    private bool disposed;

    public CheckpointPipelineSession(string checkpointPath, string textEmbeddingsPath, string? torchSharpLibraryPath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(textEmbeddingsPath);
        if (torchSharpLibraryPath is not null)
        {
            TorchSharpRuntime.Initialize(torchSharpLibraryPath);
        }
        InitializeDeviceType(DeviceType.CUDA);
        if (!cuda.is_available() || cuda.device_count() != 1)
        {
            throw new InvalidOperationException("Checkpoint pipelines require exactly one CUDA device.");
        }
        TorchSharpRuntime.UseMathSdpOnly();
        store = new CheckpointTensorStore([checkpointPath]);
        contexts = SafeTensorIndex.Open(textEmbeddingsPath);
        if (!contexts.Tensors.ContainsKey("video_context") || !contexts.Tensors.ContainsKey("audio_context"))
        {
            throw new InvalidDataException("Text embeddings must contain video_context and audio_context tensors.");
        }
        transformer = new CheckpointTransformerRuntime(store, CUDA);
        videoDecoder = new CheckpointVideoDecoder(store, CUDA);
        audioDecoder = new CheckpointAudioDecoder(store, CUDA);
        vocoder = new CheckpointVocoder(store, CUDA);
    }

    public string ModelVersion => transformer.ModelVersion;

    public CheckpointPipelineOutput Generate(PipelineMode mode, PipelineRequest request)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ValidateGeometry(request);
        using var scope = NewDisposeScope();
        manual_seed(request.Seed);
        cuda.manual_seed_all(request.Seed);
        var videoEnabled = !mode.AudioOnly;
        var audioEnabled = !mode.HdrBatch;
        var latentFrames = (request.FrameCount - 1) / 8 + 1;
        var latentHeight = request.Height / 32;
        var latentWidth = request.Width / 32;
        var videoTokens = checked(latentFrames * latentHeight * latentWidth);
        var duration = request.FrameCount / request.FramesPerSecond;
        var audioTokens = Math.Max(1, (int)Math.Ceiling(duration * 25));
        var videoLatent = videoEnabled
            ? randn([1, videoTokens, 128], dtype: ScalarType.BFloat16, device: CUDA)
            : null;
        var audioLatent = audioEnabled
            ? randn([1, audioTokens, 128], dtype: ScalarType.BFloat16, device: CUDA)
            : null;
        using var videoContext = videoEnabled ? contexts.LoadTorchTensor("video_context", CUDA) : null;
        using var audioContext = audioEnabled ? contexts.LoadTorchTensor("audio_context", CUDA) : null;
        using var videoPositions = videoEnabled
            ? VideoPositions(latentFrames, latentHeight, latentWidth, request.FramesPerSecond)
            : null;
        using var audioPositions = audioEnabled ? AudioPositions(audioTokens) : null;

        var sigmas = ResolveSigmas(request.InferenceSteps);
        for (var step = 0; step < sigmas.Length - 1; step++)
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
            if (videoEnabled)
            {
                var previous = videoLatent!;
                videoLatent = previous.add(output.Video!.mul(next - sigma)).to(ScalarType.BFloat16)
                    .MoveToOuterDisposeScope();
                previous.Dispose();
            }
            if (audioEnabled)
            {
                var previous = audioLatent!;
                audioLatent = previous.add(output.Audio!.mul(next - sigma)).to(ScalarType.BFloat16)
                    .MoveToOuterDisposeScope();
                previous.Dispose();
            }
        }

        RgbVideo? video = null;
        AudioData? audio = null;
        if (videoEnabled)
        {
            using var latent = videoLatent!.reshape(1, latentFrames, latentHeight, latentWidth, 128)
                .permute(0, 4, 1, 2, 3).contiguous();
            using var decoded = videoDecoder.Decode(latent);
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
        return new CheckpointPipelineOutput(video, audio);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        transformer.Dispose();
        store.Dispose();
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
