namespace Ltx.Media;

public static class HdrColor
{
    private static readonly float[,] AcesCgToRec2020 =
    {
        { 1.0258247f, -0.0200532f, -0.0057715f },
        { -0.0022350f, 1.0045865f, -0.0023515f },
        { -0.0050133f, -0.0252901f, 1.0303035f },
    };

    public static FloatImage ToRec2020Linear(FloatImage source, HdrColorSpace colorSpace)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(source), "HDR conversion requires at least RGB channels.");
        }
        var output = (float[])source.Pixels.Clone();
        for (var offset = 0; offset < output.Length; offset += source.Channels)
        {
            var red = output[offset];
            var green = output[offset + 1];
            var blue = output[offset + 2];
            if (colorSpace == HdrColorSpace.AcesCct)
            {
                red = DecodeAcesCct(red);
                green = DecodeAcesCct(green);
                blue = DecodeAcesCct(blue);
                colorSpace = HdrColorSpace.AcesCg;
            }
            if (colorSpace == HdrColorSpace.AcesCg)
            {
                output[offset] = Dot(AcesCgToRec2020, 0, red, green, blue);
                output[offset + 1] = Dot(AcesCgToRec2020, 1, red, green, blue);
                output[offset + 2] = Dot(AcesCgToRec2020, 2, red, green, blue);
            }
            else
            {
                output[offset] = red;
                output[offset + 1] = green;
                output[offset + 2] = blue;
            }
        }
        return new FloatImage(output, source.Width, source.Height, source.Channels);
    }

    public static RgbVideo ToneMapToSdr(IReadOnlyList<FloatImage> frames, double framesPerSecond)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("At least one HDR frame is required.", nameof(frames));
        }
        var first = frames[0];
        var pixels = new byte[checked(first.Width * first.Height * 3 * frames.Count)];
        var destination = 0;
        foreach (var frame in frames)
        {
            if (frame.Width != first.Width || frame.Height != first.Height || frame.Channels < 3)
            {
                throw new ArgumentException("HDR frame layouts must match.", nameof(frames));
            }
            for (var source = 0; source < frame.Pixels.Length; source += frame.Channels)
            {
                pixels[destination++] = ToneMap(frame.Pixels[source]);
                pixels[destination++] = ToneMap(frame.Pixels[source + 1]);
                pixels[destination++] = ToneMap(frame.Pixels[source + 2]);
            }
        }
        return new RgbVideo(pixels, first.Width, first.Height, frames.Count, framesPerSecond);
    }

    public static float EncodeHlg(float linear)
    {
        linear = MathF.Max(0, linear);
        return linear <= (1f / 12f)
            ? MathF.Sqrt(3f * linear)
            : 0.17883277f * MathF.Log(12f * linear - 0.28466892f) + 0.55991073f;
    }

    private static byte ToneMap(float linear)
    {
        var mapped = MathF.Max(0, linear) / (1f + MathF.Max(0, linear));
        var srgb = mapped <= 0.0031308f
            ? 12.92f * mapped
            : 1.055f * MathF.Pow(mapped, 1f / 2.4f) - 0.055f;
        return (byte)Math.Clamp(MathF.Round(srgb * 255f), 0, 255);
    }

    private static float DecodeAcesCct(float encoded) => encoded <= 0.15525114f
        ? (encoded - 0.07290553f) / 10.54023774f
        : MathF.Pow(2f, encoded * 17.52f - 9.72f);

    private static float Dot(float[,] matrix, int row, float red, float green, float blue) =>
        matrix[row, 0] * red + matrix[row, 1] * green + matrix[row, 2] * blue;
}
