using System.Security.Cryptography;
using Ltx.Media;

namespace Ltx.Pipelines;

public static class SeededMedia
{
    public static RgbVideo GenerateVideo(PipelineMode mode, PipelineRequest request)
    {
        ValidateVideoShape(request.Width, request.Height, request.FrameCount, request.FramesPerSecond);
        var pixels = new byte[checked(request.Width * request.Height * request.FrameCount * 3)];
        var modeIndex = PipelineMode.InScope.IndexOf(mode);
        var state = unchecked((uint)request.Seed ^ 0x9e3779b9u ^ ((uint)(modeIndex + 1) * 0x85ebca6bu));
        for (var index = 0; index < pixels.Length; index++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels[index] = (byte)(state & 0xff);
        }
        return new RgbVideo(pixels, request.Width, request.Height, request.FrameCount, request.FramesPerSecond);
    }

    public static IReadOnlyList<FloatImage> GenerateHdrFrames(PipelineMode mode, PipelineRequest request)
    {
        var video = GenerateVideo(mode, request);
        var frames = new List<FloatImage>(request.FrameCount);
        var frameBytes = video.FrameByteCount;
        for (var frameIndex = 0; frameIndex < request.FrameCount; frameIndex++)
        {
            var values = new float[request.Width * request.Height * 3];
            var source = frameIndex * frameBytes;
            for (var index = 0; index < values.Length; index++)
            {
                values[index] = video.Pixels[source + index] / 63.75f;
            }
            frames.Add(new FloatImage(values, request.Width, request.Height, 3));
        }
        return frames;
    }

    public static AudioData GenerateAudio(PipelineMode mode, PipelineRequest request, int sampleRate = 16_000)
    {
        var modeIndex = PipelineMode.InScope.IndexOf(mode);
        var sampleCount = Math.Max(1, (int)Math.Round(request.FrameCount / request.FramesPerSecond * sampleRate));
        var samples = new float[sampleCount];
        var frequency = 220 + modeIndex * 17;
        var phase = (request.Seed % 97) / 97.0;
        for (var index = 0; index < samples.Length; index++)
        {
            samples[index] = 0.2f * MathF.Sin((float)(2 * Math.PI * frequency * index / sampleRate + phase));
        }
        return new AudioData(samples, sampleRate);
    }

    public static string Sha256(RgbVideo video) => Convert.ToHexString(SHA256.HashData(video.Pixels)).ToLowerInvariant();

    private static void ValidateVideoShape(int width, int height, int frames, double framesPerSecond)
    {
        if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Video width and height must be positive multiples of 32.");
        }
        if (frames <= 0 || (frames - 1) % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frames), "Video frame count must satisfy 8k+1.");
        }
        if (framesPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(framesPerSecond));
        }
    }

    private static int IndexOf(this IReadOnlyList<PipelineMode> modes, PipelineMode mode)
    {
        for (var index = 0; index < modes.Count; index++)
        {
            if (modes[index] == mode)
            {
                return index;
            }
        }
        throw new ArgumentException("Pipeline mode is not part of the in-scope inventory.", nameof(mode));
    }
}
