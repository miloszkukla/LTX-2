using Ltx.Media;

namespace Ltx.Pipelines;

public interface IMediaPipeline
{
    PipelineMode Mode { get; }

    PipelineResult Execute(PipelineRequest request);
}

public abstract class MediaPipelineBase(PipelineMode mode) : IMediaPipeline
{
    public PipelineMode Mode { get; } = mode;

    public virtual PipelineResult Execute(PipelineRequest request) => PipelineExecution.Execute(Mode, request);
}

public sealed class A2VidPipelineTwoStage() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.a2vid_two_stage"));
public sealed class DFRPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.dfr_pipeline"));
public sealed class DistilledPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.distilled"));
public sealed class DubItPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.dubit"));
public sealed class HDRICLoraPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.hdr_ic_lora"));
public sealed class ICLoraPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.ic_lora"));
public sealed class KeyframeInterpolationPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.keyframe_interpolation"));
public sealed class RetakePipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.retake"));
public sealed class T2AOneStagePipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.t2a_one_stage"));
public sealed class TI2VidOneStagePipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.ti2vid_one_stage"));
public sealed class TI2VidTwoStagesPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.ti2vid_two_stages"));
public sealed class TI2VidTwoStagesHQPipeline() : MediaPipelineBase(PipelineMode.Resolve("ltx_pipelines.ti2vid_two_stages_hq"));

public static class PipelineRegistry
{
    public static IMediaPipeline Create(PipelineMode mode) => mode.ModuleName switch
    {
        "ltx_pipelines.a2vid_two_stage" => new A2VidPipelineTwoStage(),
        "ltx_pipelines.dfr_pipeline" => new DFRPipeline(),
        "ltx_pipelines.distilled" => new DistilledPipeline(),
        "ltx_pipelines.dubit" => new DubItPipeline(),
        "ltx_pipelines.hdr_ic_lora" => new HDRICLoraPipeline(),
        "ltx_pipelines.ic_lora" => new ICLoraPipeline(),
        "ltx_pipelines.keyframe_interpolation" => new KeyframeInterpolationPipeline(),
        "ltx_pipelines.retake" => new RetakePipeline(),
        "ltx_pipelines.t2a_one_stage" => new T2AOneStagePipeline(),
        "ltx_pipelines.ti2vid_one_stage" => new TI2VidOneStagePipeline(),
        "ltx_pipelines.ti2vid_two_stages" => new TI2VidTwoStagesPipeline(),
        "ltx_pipelines.ti2vid_two_stages_hq" => new TI2VidTwoStagesHQPipeline(),
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };
}

internal static class PipelineExecution
{
    public static PipelineResult Execute(PipelineMode mode, PipelineRequest request)
    {
        if (!request.FixtureMode)
        {
            return ExecuteCheckpoint(mode, request);
        }
        if (mode.AudioOnly)
        {
            var output = Required(request.OutputPath, "--output-path");
            WaveCodec.WritePcm16(output, SeededMedia.GenerateAudio(mode, request));
            return new PipelineResult(mode.ModuleName, [Path.GetFullPath(output)], 0, request.FramesPerSecond, true, false);
        }
        if (mode.HdrBatch)
        {
            return ExecuteHdrBatch(mode, request);
        }

        var video = CreateModeVideo(mode, request);
        var audio = CreateModeAudio(mode, request);
        var outputPath = Required(request.OutputPath, "--output-path");
        FfmpegMedia.EncodeVideo(outputPath, video, audio, hdr: request.HdrColorSpace is not null);
        return new PipelineResult(
            mode.ModuleName,
            [Path.GetFullPath(outputPath)],
            video.FrameCount,
            video.FramesPerSecond,
            audio is not null,
            request.HdrColorSpace is not null);
    }

    private static PipelineResult ExecuteCheckpoint(PipelineMode mode, PipelineRequest request)
    {
        var ownsSession = request.CheckpointSession is null;
        var session = request.CheckpointSession ?? new CheckpointPipelineSession(
            Required(request.CheckpointPath, "checkpoint path"),
            Required(request.TextEmbeddingsPath, "--text-embeddings"),
            request.TorchSharpLibraryPath);
        try
        {
            var generated = session.Generate(mode, request);
            if (mode.AudioOnly)
            {
                var output = Required(request.OutputPath, "--output-path");
                WaveCodec.WritePcm16(output, generated.Audio!);
                return new PipelineResult(
                    mode.ModuleName, [Path.GetFullPath(output)], 0, request.FramesPerSecond,
                    true, false, "native_csharp_full_checkpoint");
            }
            if (mode.HdrBatch)
            {
                var outputDirectory = Required(request.OutputDirectory, "--output-dir");
                Directory.CreateDirectory(outputDirectory);
                var stem = Path.GetFileNameWithoutExtension(Required(request.InputPath, "--input"));
                var exrDirectory = Path.Combine(outputDirectory, stem + "_exr");
                Directory.CreateDirectory(exrDirectory);
                var video = generated.Video!;
                var outputs = new List<string>();
                for (var frame = 0; frame < video.FrameCount; frame++)
                {
                    var values = new float[video.FrameByteCount];
                    for (var index = 0; index < values.Length; index++)
                    {
                        values[index] = video.Pixels[frame * video.FrameByteCount + index] / 255F;
                    }
                    var path = Path.Combine(exrDirectory, $"frame_{frame:D5}.exr");
                    OpenImageIo.WriteExr(path, new FloatImage(values, video.Width, video.Height, 3));
                    outputs.Add(Path.GetFullPath(path));
                }
                return new PipelineResult(
                    mode.ModuleName, outputs, video.FrameCount, video.FramesPerSecond,
                    false, true, "native_csharp_full_checkpoint");
            }
            var outputPath = Required(request.OutputPath, "--output-path");
            FfmpegMedia.EncodeVideo(outputPath, generated.Video!, generated.Audio, hdr: request.HdrColorSpace is not null);
            return new PipelineResult(
                mode.ModuleName, [Path.GetFullPath(outputPath)], generated.Video!.FrameCount,
                generated.Video.FramesPerSecond, generated.Audio is not null,
                request.HdrColorSpace is not null, "native_csharp_full_checkpoint");
        }
        finally
        {
            if (ownsSession) session.Dispose();
        }
    }

    private static PipelineResult ExecuteHdrBatch(PipelineMode mode, PipelineRequest request)
    {
        var input = Required(request.InputPath, "--input");
        if (!File.Exists(input))
        {
            throw new FileNotFoundException("HDR IC-LoRA input video does not exist.", input);
        }
        var outputDirectory = Required(request.OutputDirectory, "--output-dir");
        Directory.CreateDirectory(outputDirectory);
        var stem = Path.GetFileNameWithoutExtension(input);
        var exrDirectory = Path.Combine(outputDirectory, stem + "_exr");
        Directory.CreateDirectory(exrDirectory);
        var frames = SeededMedia.GenerateHdrFrames(mode, request);
        var outputs = new List<string>(frames.Count + 1);
        for (var index = 0; index < frames.Count; index++)
        {
            var path = Path.Combine(exrDirectory, $"frame_{index:D5}.exr");
            OpenImageIo.WriteExr(path, frames[index]);
            outputs.Add(Path.GetFullPath(path));
        }
        if (!request.SkipPreview)
        {
            var preview = Path.Combine(outputDirectory, stem + ".mov");
            FfmpegMedia.EncodeVideo(preview, HdrColor.ToneMapToSdr(frames, request.FramesPerSecond),
                SeededMedia.GenerateAudio(mode, request), proResPreview: true);
            outputs.Add(Path.GetFullPath(preview));
        }
        return new PipelineResult(mode.ModuleName, outputs, frames.Count, request.FramesPerSecond, !request.SkipPreview, true);
    }

    private static RgbVideo CreateModeVideo(PipelineMode mode, PipelineRequest request)
    {
        if (mode.ModuleName == "ltx_pipelines.retake")
        {
            var sourcePath = Required(request.VideoPath, "--video-path");
            var source = Directory.Exists(sourcePath)
                ? DecodeExrSequence(sourcePath, request.FramesPerSecond)
                : FfmpegMedia.DecodeVideo(sourcePath);
            if (request.StartTime >= request.EndTime)
            {
                throw new ArgumentException("start_time must be less than end_time");
            }
            if ((source.FrameCount - 1) % 8 != 0 || source.Width % 32 != 0 || source.Height % 32 != 0)
            {
                throw new ArgumentException("Retake input must use 8k+1 frames and dimensions divisible by 32.");
            }
            var replacementRequest = request with
            {
                Width = source.Width,
                Height = source.Height,
                FrameCount = source.FrameCount,
                FramesPerSecond = source.FramesPerSecond,
            };
            var replacement = SeededMedia.GenerateVideo(mode, replacementRequest);
            var pixels = (byte[])source.Pixels.Clone();
            var firstFrame = Math.Clamp((int)Math.Floor(request.StartTime * source.FramesPerSecond), 0, source.FrameCount - 1);
            var lastFrame = Math.Clamp((int)Math.Ceiling(request.EndTime * source.FramesPerSecond), firstFrame + 1, source.FrameCount);
            Buffer.BlockCopy(replacement.Pixels, firstFrame * source.FrameByteCount, pixels,
                firstFrame * source.FrameByteCount, (lastFrame - firstFrame) * source.FrameByteCount);
            return source with { Pixels = pixels };
        }
        if (mode.ModuleName == "ltx_pipelines.dubit")
        {
            var reference = FfmpegMedia.DecodeVideo(Required(request.ReferenceVideoPath, "--reference-video"));
            var snappedFrames = ((reference.FrameCount - 1) / 8) * 8 + 1;
            return SeededMedia.GenerateVideo(mode, request with
            {
                FrameCount = snappedFrames,
                FramesPerSecond = reference.FramesPerSecond,
            });
        }
        return SeededMedia.GenerateVideo(mode, request);
    }

    private static AudioData? CreateModeAudio(PipelineMode mode, PipelineRequest request)
    {
        if (mode.ModuleName == "ltx_pipelines.a2vid_two_stage")
        {
            return FfmpegMedia.DecodeAudio(Required(request.AudioPath, "--audio-path"));
        }
        if (mode.ModuleName == "ltx_pipelines.dubit")
        {
            return FfmpegMedia.DecodeAudio(Required(request.ReferenceVideoPath, "--reference-video"));
        }
        if (mode.ModuleName == "ltx_pipelines.retake" && request.VideoPath is not null && File.Exists(request.VideoPath))
        {
            var probe = FfmpegMedia.ProbeVideo(request.VideoPath);
            return probe.HasAudio ? FfmpegMedia.DecodeAudio(request.VideoPath) : SeededMedia.GenerateAudio(mode, request);
        }
        return SeededMedia.GenerateAudio(mode, request);
    }

    private static RgbVideo DecodeExrSequence(string directory, double framesPerSecond)
    {
        var paths = Directory.EnumerateFiles(directory, "*.exr").Order(StringComparer.Ordinal).ToArray();
        if (paths.Length == 0)
        {
            throw new InvalidDataException("EXR sequence directory is empty.");
        }
        var frames = paths.Select(OpenImageIo.ReadExr).ToArray();
        return HdrColor.ToneMapToSdr(frames, framesPerSecond);
    }

    private static string Required(string? value, string flag) => string.IsNullOrWhiteSpace(value)
        ? throw new ArgumentException($"Missing required argument {flag}.")
        : value;
}
