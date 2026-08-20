using System.Text.Json;
using Ltx.Media;
using Ltx.Pipelines;

if (args.Length != 1)
{
    Console.Error.WriteLine("usage: Ltx.MediaPipelineTests <fixture-dir>");
    return 2;
}

var fixture = Path.GetFullPath(args[0]);
using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture, "manifest.json")));
var root = document.RootElement;
if (root.GetProperty("schema_version").GetInt32() != 1 ||
    root.GetProperty("fixture_revision").GetString() != "m5-media-pipelines-v1" ||
    root.GetProperty("seed").GetInt32() != 20260820)
{
    throw new InvalidDataException("M5 fixture version does not match the pinned oracle.");
}

var shape = root.GetProperty("shape");
var width = shape.GetProperty("width").GetInt32();
var height = shape.GetProperty("height").GetInt32();
var frames = shape.GetProperty("frames").GetInt32();
var fps = shape.GetProperty("fps").GetDouble();
var seed = root.GetProperty("seed").GetInt32();
var mediaChecks = 0;

var info = FfmpegMedia.ProbeVideo(Path.Combine(fixture, "input.mp4"));
Assert(info.Width == width && info.Height == height && info.FrameCount == frames &&
       Math.Abs(info.FramesPerSecond - fps) < 1e-9 && info.HasAudio, "FFmpeg video probe");
mediaChecks++;

var decodedVideo = FfmpegMedia.DecodeVideo(Path.Combine(fixture, "input.mp4"));
Assert(decodedVideo.Width == width && decodedVideo.Height == height && decodedVideo.FrameCount == frames, "video decode");
mediaChecks++;

var muxPath = Path.Combine(Path.GetTempPath(), $"ltx-mux-frame-count-{Guid.NewGuid():N}.mp4");
try
{
    var muxVideo = new RgbVideo(new byte[32 * 32 * 3 * 9], 32, 32, 9, 8);
    var shorterAudio = new AudioData(new float[8_000], 8_000);
    FfmpegMedia.EncodeVideo(muxPath, muxVideo, shorterAudio);
    var muxInfo = FfmpegMedia.ProbeVideo(muxPath);
    Assert(muxInfo.FrameCount == muxVideo.FrameCount, "audio must not truncate muxed video frames");
}
finally
{
    if (File.Exists(muxPath)) File.Delete(muxPath);
}
mediaChecks++;

var expectedAudio = root.GetProperty("audio");
var wave = WaveCodec.ReadPcm16(Path.Combine(fixture, "input.wav"));
Assert(wave.SampleRate == expectedAudio.GetProperty("sample_rate").GetInt32() &&
       wave.Samples.Length == expectedAudio.GetProperty("sample_count").GetInt32(), "WAVE layout");
AssertNear(wave.Samples.Take(8).ToArray(), Floats(expectedAudio.GetProperty("first_samples")), 1e-4f, 1e-5f, "WAVE samples");
mediaChecks++;

var containerAudio = FfmpegMedia.DecodeAudio(Path.Combine(fixture, "input.mp4"));
Assert(containerAudio.SampleRate == 16_000 && containerAudio.Samples.Length > 16_000, "container audio decode");
mediaChecks++;

var still = FfmpegMedia.DecodeStill(Path.Combine(fixture, "input.png"));
Assert(still.Width == width && still.Height == height && still.FrameCount == 1, "still decode");
mediaChecks++;

Assert(OpenImageIo.RuntimeVersion.StartsWith("2.4.", StringComparison.Ordinal), "OpenImageIO native bridge version");
mediaChecks++;

var exr = OpenImageIo.ReadExr(Path.Combine(fixture, "input.exr"));
Assert(exr.Width == width && exr.Height == height && exr.Channels == 3, "EXR layout");
AssertNear(exr.Pixels.Take(3).ToArray(), [0.25f, 0.5f, 2f], 1e-4f, 1e-5f, "EXR pixels");
mediaChecks++;

var roundTripPath = Path.Combine(Path.GetTempPath(), $"ltx-m5-exr-{Guid.NewGuid():N}.exr");
try
{
    OpenImageIo.WriteExr(roundTripPath, exr);
    var roundTrip = OpenImageIo.ReadExr(roundTripPath);
    AssertNear(roundTrip.Pixels.Take(64).ToArray(), exr.Pixels.Take(64).ToArray(), 1e-4f, 1e-5f, "EXR round trip");
}
finally
{
    if (File.Exists(roundTripPath)) File.Delete(roundTripPath);
}
mediaChecks++;

var toneMapped = HdrColor.ToneMapToSdr([exr], fps);
Assert(toneMapped.Width == width && toneMapped.Height == height && toneMapped.Pixels.Any(value => value > 0), "HDR tone map");
AssertNear([HdrColor.EncodeHlg(1f / 12f)], [0.5f], 1e-4f, 1e-5f, "HLG transfer");
mediaChecks++;

var manifestModes = root.GetProperty("pipeline_modes").EnumerateArray().ToArray();
Assert(manifestModes.Length == 12 && PipelineMode.InScope.Count == 12, "pipeline inventory count");
var pipelineChecks = 0;
for (var index = 0; index < manifestModes.Length; index++)
{
    var expected = manifestModes[index];
    var mode = PipelineMode.InScope[index];
    Assert(mode.ModuleName == expected.GetProperty("module").GetString() &&
           mode.CommandName == expected.GetProperty("command").GetString() &&
           mode.PipelineName == expected.GetProperty("pipeline").GetString(), $"pipeline inventory {mode.ModuleName}");
    var pipeline = PipelineRegistry.Create(mode);
    Assert(pipeline.GetType().Name == mode.PipelineName, $"pipeline type {mode.ModuleName}");
    var request = new PipelineRequest
    {
        Prompt = "small seeded M5 fixture",
        Seed = seed,
        Width = width,
        Height = height,
        FrameCount = frames,
        FramesPerSecond = fps,
    };
    var preview = SeededMedia.GenerateVideo(mode, request);
    Assert(SeededMedia.Sha256(preview) == expected.GetProperty("preview_sha256").GetString(),
        $"seeded video {mode.ModuleName}");
    var audio = SeededMedia.GenerateAudio(mode, request);
    AssertNear(audio.Samples.Take(8).ToArray(), Floats(expected.GetProperty("audio_first_samples")),
        1e-4f, 1e-5f, $"seeded audio {mode.ModuleName}");
    pipelineChecks++;
}

var total = mediaChecks + pipelineChecks;
Console.WriteLine($"M5 media/pipeline fixtures: {total} passed, 0 failed; media={mediaChecks}, pipelines={pipelineChecks}");
return 0;

static float[] Floats(JsonElement element) => element.EnumerateArray().Select(value => value.GetSingle()).ToArray();

static void Assert(bool condition, string name)
{
    if (!condition) throw new InvalidDataException($"{name} failed.");
}

static void AssertNear(
    IReadOnlyList<float> actual,
    IReadOnlyList<float> expected,
    float relativeTolerance,
    float absoluteTolerance,
    string name)
{
    Assert(actual.Count == expected.Count, name + " length");
    for (var index = 0; index < actual.Count; index++)
    {
        var difference = MathF.Abs(actual[index] - expected[index]);
        var limit = absoluteTolerance + relativeTolerance * MathF.Abs(expected[index]);
        if (difference > limit)
        {
            throw new InvalidDataException(
                $"{name}[{index}] actual={actual[index]:g9} expected={expected[index]:g9} diff={difference:g9} limit={limit:g9}");
        }
    }
}
