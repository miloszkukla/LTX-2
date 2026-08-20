namespace Ltx.Media;

public enum HdrColorSpace
{
    SrgbLinear,
    AcesCg,
    AcesCct,
}

public sealed record AudioData(float[] Samples, int SampleRate, int Channels = 1)
{
    public double DurationSeconds => Samples.Length / (double)(SampleRate * Channels);
}

public sealed record VideoInfo(
    int Width,
    int Height,
    int FrameCount,
    double FramesPerSecond,
    bool HasAudio,
    string PixelFormat,
    string? ColorPrimaries,
    string? ColorTransfer);

public sealed record RgbVideo(byte[] Pixels, int Width, int Height, int FrameCount, double FramesPerSecond)
{
    public int FrameByteCount => checked(Width * Height * 3);
}

public sealed record FloatImage(float[] Pixels, int Width, int Height, int Channels)
{
    public int PixelCount => checked(Width * Height);
}
