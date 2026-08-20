using System.Globalization;
using System.Text.Json;

namespace Ltx.Media;

public static class FfmpegMedia
{
    public static VideoInfo ProbeVideo(string path)
    {
        EnsureFile(path);
        var json = ToolProcess.CaptureText("ffprobe",
        [
            "-v", "error",
            "-show_entries", "stream=index,codec_type,width,height,nb_frames,avg_frame_rate,pix_fmt,color_primaries,color_transfer",
            "-of", "json",
            path,
        ]);
        using var document = JsonDocument.Parse(json);
        var streams = document.RootElement.GetProperty("streams").EnumerateArray().ToArray();
        var video = streams.FirstOrDefault(element =>
            element.TryGetProperty("codec_type", out var type) && type.GetString() == "video");
        if (video.ValueKind == JsonValueKind.Undefined)
        {
            throw new InvalidDataException($"'{path}' has no video stream.");
        }

        var frameRate = ParseRate(GetString(video, "avg_frame_rate") ?? "0/1");
        var frames = ParseInteger(GetString(video, "nb_frames"));
        if (frames <= 0)
        {
            var raw = ToolProcess.CaptureText("ffprobe",
            [
                "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", path,
            ]).Trim();
            frames = ParseInteger(raw);
        }

        return new VideoInfo(
            video.GetProperty("width").GetInt32(),
            video.GetProperty("height").GetInt32(),
            frames,
            frameRate,
            streams.Any(element => element.TryGetProperty("codec_type", out var type) && type.GetString() == "audio"),
            GetString(video, "pix_fmt") ?? "unknown",
            GetString(video, "color_primaries"),
            GetString(video, "color_transfer"));
    }

    public static RgbVideo DecodeVideo(string path, int? maximumFrames = null)
    {
        var info = ProbeVideo(path);
        var arguments = new List<string>
        {
            "-hide_banner", "-loglevel", "error", "-i", path,
            "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
        };
        if (maximumFrames is > 0)
        {
            arguments.AddRange(["-frames:v", maximumFrames.Value.ToString(CultureInfo.InvariantCulture)]);
        }
        arguments.Add("pipe:1");
        var pixels = ToolProcess.CaptureBytes("ffmpeg", arguments);
        var frameBytes = checked(info.Width * info.Height * 3);
        if (pixels.Length == 0 || pixels.Length % frameBytes != 0)
        {
            throw new InvalidDataException("FFmpeg returned a partial RGB frame sequence.");
        }
        return new RgbVideo(pixels, info.Width, info.Height, pixels.Length / frameBytes, info.FramesPerSecond);
    }

    public static RgbVideo DecodeStill(string path)
    {
        EnsureFile(path);
        var dimensions = ToolProcess.CaptureText("ffprobe",
        [
            "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", path,
        ]).Trim().Split('x');
        if (dimensions.Length != 2 || !int.TryParse(dimensions[0], out var width) ||
            !int.TryParse(dimensions[1], out var height))
        {
            throw new InvalidDataException("Could not determine still-image dimensions.");
        }
        var pixels = ToolProcess.CaptureBytes("ffmpeg",
        [
            "-hide_banner", "-loglevel", "error", "-i", path, "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]);
        if (pixels.Length != checked(width * height * 3))
        {
            throw new InvalidDataException("FFmpeg returned an incomplete still image.");
        }
        return new RgbVideo(pixels, width, height, 1, 1);
    }

    public static AudioData DecodeAudio(string path, int sampleRate = 16_000, int channels = 1)
    {
        EnsureFile(path);
        var raw = ToolProcess.CaptureBytes("ffmpeg",
        [
            "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-map", "0:a:0",
            "-ac", channels.ToString(CultureInfo.InvariantCulture),
            "-ar", sampleRate.ToString(CultureInfo.InvariantCulture),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ]);
        if (raw.Length == 0 || raw.Length % sizeof(float) != 0)
        {
            throw new InvalidDataException("FFmpeg returned an incomplete audio stream.");
        }
        var samples = new float[raw.Length / sizeof(float)];
        for (var index = 0; index < samples.Length; index++)
        {
            samples[index] = BitConverter.ToSingle(raw, index * sizeof(float));
        }
        return new AudioData(samples, sampleRate, channels);
    }

    public static void EncodeVideo(
        string outputPath,
        RgbVideo video,
        AudioData? audio = null,
        bool hdr = false,
        bool proResPreview = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(video);
        if (video.Width <= 0 || video.Height <= 0 || video.FrameCount <= 0 || video.FramesPerSecond <= 0 ||
            video.Pixels.Length != checked(video.Width * video.Height * 3 * video.FrameCount))
        {
            throw new ArgumentOutOfRangeException(nameof(video), "RGB video layout is inconsistent.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        string? temporaryWave = null;
        try
        {
            if (audio is not null)
            {
                temporaryWave = Path.Combine(Path.GetTempPath(), $"ltx-media-{Guid.NewGuid():N}.wav");
                WaveCodec.WritePcm16(temporaryWave, audio);
            }

            var arguments = new List<string>
            {
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", $"{video.Width}x{video.Height}",
                "-framerate", video.FramesPerSecond.ToString("0.########", CultureInfo.InvariantCulture),
                "-i", "pipe:0",
            };
            if (temporaryWave is not null)
            {
                arguments.AddRange(["-i", temporaryWave, "-map", "0:v:0", "-map", "1:a:0"]);
            }
            else
            {
                arguments.AddRange(["-map", "0:v:0"]);
            }

            if (proResPreview)
            {
                arguments.AddRange(["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]);
            }
            else
            {
                arguments.AddRange(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"]);
            }
            if (hdr)
            {
                arguments.AddRange([
                    "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
                ]);
            }
            if (temporaryWave is not null)
            {
                arguments.AddRange(proResPreview
                    ? ["-c:a", "pcm_s16le", "-shortest"]
                    : ["-c:a", "aac", "-b:a", "96k", "-shortest"]);
            }
            arguments.AddRange(["-map_metadata", "-1", "-fflags", "+bitexact", outputPath]);

            using var input = new MemoryStream(video.Pixels, writable: false);
            ToolProcess.Run("ffmpeg", arguments, input, output: null);
        }
        finally
        {
            if (temporaryWave is not null && File.Exists(temporaryWave))
            {
                File.Delete(temporaryWave);
            }
        }
    }

    private static string? GetString(JsonElement element, string name) =>
        element.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static int ParseInteger(string? value) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result) ? result : 0;

    private static double ParseRate(string value)
    {
        var parts = value.Split('/');
        if (parts.Length == 2 &&
            double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var numerator) &&
            double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var denominator) &&
            denominator != 0)
        {
            return numerator / denominator;
        }
        return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var scalar) ? scalar : 0;
    }

    private static void EnsureFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Media input does not exist.", path);
        }
    }
}
