using Ltx.Media;

namespace Ltx.Pipelines;

public sealed record PipelineRequest
{
    public required string Prompt { get; init; }
    public required int Seed { get; init; }
    public required int Width { get; init; }
    public required int Height { get; init; }
    public required int FrameCount { get; init; }
    public required double FramesPerSecond { get; init; }
    public string? OutputPath { get; init; }
    public string? OutputDirectory { get; init; }
    public string? AudioPath { get; init; }
    public string? ReferenceVideoPath { get; init; }
    public string? VideoPath { get; init; }
    public string? ConditioningPath { get; init; }
    public string? InputPath { get; init; }
    public double StartTime { get; init; }
    public double EndTime { get; init; }
    public HdrColorSpace? HdrColorSpace { get; init; }
    public bool SkipPreview { get; init; }
    public bool FixtureMode { get; init; }
    public string? CheckpointPath { get; init; }
    public string? SpatialUpsamplerPath { get; init; }
    public string OffloadMode { get; init; } = "none";
    public string? TextEmbeddingsPath { get; init; }
    public string? TorchSharpLibraryPath { get; init; }
    public string? LatentDiagnosticsPath { get; init; }
    public int InferenceSteps { get; init; } = 1;
    public CheckpointPipelineSession? CheckpointSession { get; init; }
}

public sealed record PipelineResult(
    string Mode,
    IReadOnlyList<string> OutputPaths,
    int FrameCount,
    double FramesPerSecond,
    bool HasAudio,
    bool IsHdr,
    string Execution = "fixture_seeded");

public sealed record PipelineMode(
    string ModuleName,
    string CommandName,
    string PipelineName,
    bool AudioOnly = false,
    bool HdrBatch = false)
{
    public static IReadOnlyList<PipelineMode> InScope { get; } =
    [
        new("ltx_pipelines.a2vid_two_stage", "a2vid-two-stage", nameof(A2VidPipelineTwoStage)),
        new("ltx_pipelines.dfr_pipeline", "dfr-pipeline", nameof(DFRPipeline)),
        new("ltx_pipelines.distilled", "distilled", nameof(DistilledPipeline)),
        new("ltx_pipelines.dubit", "dubit", nameof(DubItPipeline)),
        new("ltx_pipelines.hdr_ic_lora", "hdr-ic-lora", nameof(HDRICLoraPipeline), HdrBatch: true),
        new("ltx_pipelines.ic_lora", "ic-lora", nameof(ICLoraPipeline)),
        new("ltx_pipelines.keyframe_interpolation", "keyframe-interpolation", nameof(KeyframeInterpolationPipeline)),
        new("ltx_pipelines.retake", "retake", nameof(RetakePipeline)),
        new("ltx_pipelines.t2a_one_stage", "t2a-one-stage", nameof(T2AOneStagePipeline), AudioOnly: true),
        new("ltx_pipelines.ti2vid_one_stage", "ti2vid-one-stage", nameof(TI2VidOneStagePipeline)),
        new("ltx_pipelines.ti2vid_two_stages", "ti2vid-two-stages", nameof(TI2VidTwoStagesPipeline)),
        new("ltx_pipelines.ti2vid_two_stages_hq", "ti2vid-two-stages-hq", nameof(TI2VidTwoStagesHQPipeline)),
    ];

    public static PipelineMode Resolve(string value)
    {
        var match = InScope.FirstOrDefault(mode =>
            string.Equals(mode.CommandName, value, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(mode.ModuleName, value, StringComparison.Ordinal) ||
            string.Equals(mode.PipelineName, value, StringComparison.Ordinal));
        return match ?? throw new ArgumentException($"Unknown pipeline mode '{value}'.", nameof(value));
    }
}
