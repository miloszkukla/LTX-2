using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Ltx.Media;
using Ltx.Pipelines;

if (args.Length != 5)
{
    Console.Error.WriteLine(
        "usage: Ltx.RealPipelineTests <checkpoint> <contexts> <libLibTorchSharp.so> <output-dir> <report.json>");
    return 2;
}

var checkpoint = Path.GetFullPath(args[0]);
var contexts = Path.GetFullPath(args[1]);
var outputDirectory = Path.GetFullPath(args[3]);
var reportPath = Path.GetFullPath(args[4]);
Directory.CreateDirectory(outputDirectory);
var stopwatch = Stopwatch.StartNew();
var results = new List<object>();
var passed = 0;
using var session = new CheckpointPipelineSession(checkpoint, contexts, args[2]);
foreach (var mode in PipelineMode.InScope)
{
    var modeDirectory = Path.Combine(outputDirectory, mode.CommandName);
    Directory.CreateDirectory(modeDirectory);
    var request = new PipelineRequest
    {
        Prompt = "M7A full-checkpoint pipeline parity",
        Seed = 20260820,
        Width = 32,
        Height = 32,
        FrameCount = 1,
        FramesPerSecond = 8,
        InferenceSteps = 1,
        CheckpointSession = session,
        CheckpointPath = checkpoint,
        TextEmbeddingsPath = contexts,
        OutputPath = Path.Combine(modeDirectory, mode.AudioOnly ? "output.wav" : "output.mp4"),
        OutputDirectory = modeDirectory,
        InputPath = Path.Combine(modeDirectory, "source.mp4"),
        AudioPath = Path.Combine(modeDirectory, "source.wav"),
        ReferenceVideoPath = Path.Combine(modeDirectory, "reference.mp4"),
        VideoPath = Path.Combine(modeDirectory, "source.mp4"),
        ConditioningPath = Path.Combine(modeDirectory, "conditioning.mp4"),
        StartTime = 0,
        EndTime = 0.125,
        HdrColorSpace = mode.HdrBatch ? HdrColorSpace.SrgbLinear : null,
        SkipPreview = true,
    };
    var result = PipelineRegistry.Create(mode).Execute(request);
    if (result.Execution != "native_csharp_full_checkpoint" || result.OutputPaths.Count == 0 ||
        result.OutputPaths.Any(path => !File.Exists(path) || new FileInfo(path).Length == 0))
    {
        throw new InvalidDataException($"{mode.ModuleName} did not produce checkpoint-backed output.");
    }
    if (mode.AudioOnly)
    {
        var audio = WaveCodec.ReadPcm16(result.OutputPaths[0]);
        if (audio.SampleRate != 48_000 || audio.Channels != 2 || audio.Samples.Length == 0 ||
            audio.Samples.All(sample => sample == 0))
        {
            throw new InvalidDataException($"{mode.ModuleName} waveform validation failed.");
        }
    }
    else if (mode.HdrBatch)
    {
        var frame = OpenImageIo.ReadExr(result.OutputPaths[0]);
        if (frame.Width != 32 || frame.Height != 32 || frame.Pixels.All(pixel => pixel == 0))
        {
            throw new InvalidDataException($"{mode.ModuleName} EXR validation failed.");
        }
    }
    else
    {
        var video = FfmpegMedia.ProbeVideo(result.OutputPaths[0]);
        if (video.Width != 32 || video.Height != 32 || video.FrameCount != 1 || !video.HasAudio)
        {
            throw new InvalidDataException($"{mode.ModuleName} MP4 validation failed.");
        }
    }
    results.Add(new
    {
        module = mode.ModuleName,
        pipeline = mode.PipelineName,
        execution = result.Execution,
        output_count = result.OutputPaths.Count,
        output_sha256 = result.OutputPaths.Select(Sha256).ToArray(),
        result = "pass",
    });
    passed++;
    Console.WriteLine($"M7A real pipeline {passed}/12: {mode.ModuleName} pass");
}

stopwatch.Stop();
var report = new
{
    schema_version = 1,
    result = "pass",
    checkpoint_model_version = session.ModelVersion,
    execution = "native_csharp_torchsharp_cuda_full_checkpoint",
    python_runtime_calls = 0,
    modes = results,
    test_totals = new { passed, failed = 0, skipped = 0 },
    elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
    secrets_recorded = false,
};
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"M7A real checkpoint pipelines: {passed} passed, 0 failed in {stopwatch.Elapsed.TotalSeconds:F1}s");
return 0;

static string Sha256(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
