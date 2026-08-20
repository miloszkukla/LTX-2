using System.Text.Json;
using Ltx.Core;
using Ltx.Media;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length == 6 && args[0] == "transformer")
{
    TorchSharpRuntime.Initialize(args[4]);
    InitializeDeviceType(DeviceType.CUDA);
    TorchSharpRuntime.UsePythonSdpPriority();
    if (!cuda.is_available() || cuda.device_count() != 1)
    {
        throw new InvalidOperationException("M7C transformer diagnostic requires exactly one CUDA device.");
    }

    using var transformerScope = NewDisposeScope();
    using var transformerStore = new CheckpointTensorStore([args[1]]);
    var stageFixture = SafeTensorIndex.Open(args[2]);
    var contextFixture = SafeTensorIndex.Open(args[3]);
    using var videoInput = stageFixture.LoadTorchTensor("stage1_step0_video_input", CUDA);
    using var audioInput = stageFixture.LoadTorchTensor("stage1_step0_audio_input", CUDA);
    using var videoExpected = stageFixture.LoadTorchTensor("stage1_step0_video_velocity", CUDA);
    using var audioExpected = stageFixture.LoadTorchTensor("stage1_step0_audio_velocity", CUDA);
    using var videoContext = contextFixture.LoadTorchTensor("video_context", CUDA);
    using var audioContext = contextFixture.LoadTorchTensor("audio_context", CUDA);
    using var sigma = tensor([1F], dtype: ScalarType.Float32, device: CUDA);
    using var videoTimesteps = full([1, 31 * 16 * 24], 1F, dtype: ScalarType.Float32, device: CUDA);
    using var audioTimesteps = full([1, 251], 1F, dtype: ScalarType.Float32, device: CUDA);
    using var videoPositions = VideoPositions(31, 16, 24, 24);
    using var audioPositions = AudioPositions(251);
    using var runtime = new CheckpointTransformerRuntime(transformerStore, CUDA, cacheWeights: false);
    var componentComparisons = new Dictionary<string, object>(StringComparer.Ordinal);

    void RecordComponent(string name, Tensor actualTensor)
    {
        var fixtureName = $"stage1_step0_block0_{name}";
        if (!stageFixture.Tensors.ContainsKey(fixtureName)) return;
        using var expectedTensor = stageFixture.LoadTorchTensor(fixtureName, CUDA);
        var metrics = Compare(actualTensor, expectedTensor);
        componentComparisons[name] = new
        {
            elements = metrics.Elements,
            max_absolute_error = metrics.MaximumAbsoluteError,
            rmse = metrics.Rmse,
            cosine = metrics.Cosine,
            exact_fraction = metrics.ExactFraction,
        };
    }

    var tensorNames = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.diffusion_model.transformer_blocks.0.attn1"] = "attn1",
        ["model.diffusion_model.transformer_blocks.0.attn1.to_q"] = "attn1_to_q",
        ["model.diffusion_model.transformer_blocks.0.attn1.to_k"] = "attn1_to_k",
        ["model.diffusion_model.transformer_blocks.0.attn1.q_norm"] = "attn1_q_norm",
        ["model.diffusion_model.transformer_blocks.0.attn1.k_norm"] = "attn1_k_norm",
        ["model.diffusion_model.transformer_blocks.0.attn1.to_v"] = "attn1_to_v",
        ["model.diffusion_model.transformer_blocks.0.attn1.q_rope"] = "attn1_q_rope",
        ["model.diffusion_model.transformer_blocks.0.attn1.k_rope"] = "attn1_k_rope",
        ["model.diffusion_model.transformer_blocks.0.attn1.attended"] = "attn1_attended",
        ["model.diffusion_model.transformer_blocks.0.attn2"] = "attn2",
        ["model.diffusion_model.transformer_blocks.0.audio_attn1"] = "audio_attn1",
        ["model.diffusion_model.transformer_blocks.0.audio_attn2"] = "audio_attn2",
        ["model.diffusion_model.transformer_blocks.0.audio_to_video_attn"] = "audio_to_video_attn",
        ["model.diffusion_model.transformer_blocks.0.video_to_audio_attn"] = "video_to_audio_attn",
        ["model.diffusion_model.transformer_blocks.0.ff"] = "ff",
        ["model.diffusion_model.transformer_blocks.0.audio_ff"] = "audio_ff",
    };
    using var actual = runtime.Forward(
        new CheckpointTransformerInput(videoInput, videoContext, null, videoTimesteps, sigma, videoPositions),
        new CheckpointTransformerInput(audioInput, audioContext, null, audioTimesteps, sigma, audioPositions),
        observeBlock: (layer, videoState, audioState) =>
        {
            if (layer == -1)
            {
                RecordComponent("video_input", videoState!);
                RecordComponent("audio_input", audioState!);
            }
            else if (layer == 0)
            {
                RecordComponent("video_output", videoState!);
                RecordComponent("audio_output", audioState!);
            }
        },
        observeTensor: (name, tensorValue) =>
        {
            if (tensorNames.TryGetValue(name, out var resolvedName))
            {
                RecordComponent(resolvedName, tensorValue);
            }
        });
    var videoMetrics = Compare(actual.Video!, videoExpected);
    var audioMetrics = Compare(actual.Audio!, audioExpected);
    var report = new
    {
        schema_version = 1,
        fixture_revision = stageFixture.Metadata.GetValueOrDefault("fixture_revision"),
        execution = "native_csharp_streamed_production_geometry_transformer",
        python_runtime_calls = 0,
        block0 = componentComparisons,
        video = new
        {
            elements = videoMetrics.Elements,
            max_absolute_error = videoMetrics.MaximumAbsoluteError,
            rmse = videoMetrics.Rmse,
            cosine = videoMetrics.Cosine,
            exact_fraction = videoMetrics.ExactFraction,
        },
        audio = new
        {
            elements = audioMetrics.Elements,
            max_absolute_error = audioMetrics.MaximumAbsoluteError,
            rmse = audioMetrics.Rmse,
            cosine = audioMetrics.Cosine,
            exact_fraction = audioMetrics.ExactFraction,
        },
    };
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[5]))!);
    File.WriteAllText(args[5], JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
    Console.WriteLine(
        $"M7C production transformer: video cosine={videoMetrics.Cosine:G9}, rmse={videoMetrics.Rmse:G9}; " +
        $"audio cosine={audioMetrics.Cosine:G9}, rmse={audioMetrics.Rmse:G9}");
    return 0;
}

if (args.Length == 3 && args[0] == "rng")
{
    TorchSharpRuntime.Initialize(args[1]);
    InitializeDeviceType(DeviceType.CUDA);
    manual_seed(20260821);
    cuda.manual_seed_all(20260821);
    using var rngScope = NewDisposeScope();
    using var video = randn([1, 31 * 16 * 24, 128], dtype: ScalarType.BFloat16, device: CUDA);
    using var audio = randn([1, 251, 128], dtype: ScalarType.BFloat16, device: CUDA);
    using var channelFirstVideo = video.reshape(1, 31, 16, 24, 128).permute(0, 4, 1, 2, 3).contiguous();
    using var channelFirstAudio = audio.reshape(1, 251, 8, 16).permute(0, 2, 1, 3).contiguous();
    new SafeTensorFile(
        new Dictionary<string, SafeTensor>(StringComparer.Ordinal)
        {
            ["video"] = SafeTensor.FromTorchTensor(channelFirstVideo),
            ["audio"] = SafeTensor.FromTorchTensor(channelFirstAudio),
        },
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["fixture_revision"] = "m7c-csharp-initial-noise-v1",
        }).Save(args[2]);
    Console.WriteLine("M7C C# initial noise: recorded");
    return 0;
}

if (args.Length != 4)
{
    Console.Error.WriteLine(
        "usage: Ltx.VideoDecoderTests <checkpoint> <stage-latents.safetensors> <libLibTorchSharp.so> <preview.mp4>");
    return 2;
}

TorchSharpRuntime.Initialize(args[2]);
InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available() || cuda.device_count() != 1)
{
    throw new InvalidOperationException("M7C video decoder test requires exactly one CUDA device.");
}

using var scope = NewDisposeScope();
using var store = new CheckpointTensorStore([args[0]]);
var fixture = SafeTensorIndex.Open(args[1]);
if (fixture.Metadata.GetValueOrDefault("fixture_revision") != "m7c-python-two-stage-latents-v1")
{
    throw new InvalidDataException("M7C stage-latent fixture revision is invalid.");
}
using var latent = fixture.LoadTorchTensor("stage2_video", CUDA);
var latentStats = Stats(latent);
var decoder = new CheckpointVideoDecoder(store, CUDA);
using var decoded = decoder.DecodeTiled(latent);
if (!decoded.shape.SequenceEqual([1L, 3L, 241L, 1024L, 1536L]) ||
    decoded.isnan().any().item<bool>() || decoded.isinf().any().item<bool>())
{
    throw new InvalidDataException("C# tiled video decode has invalid shape or non-finite output.");
}
var decodedStats = Stats(decoded);
using var frame = decoded.narrow(2, 120, 1).to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
var values = frame.data<float>().ToArray();
var pixels = new byte[1024 * 1536 * 3];
for (var y = 0; y < 1024; y++)
for (var x = 0; x < 1536; x++)
for (var channel = 0; channel < 3; channel++)
{
    var source = (channel * 1024 + y) * 1536 + x;
    var target = (y * 1536 + x) * 3 + channel;
    pixels[target] = (byte)Math.Clamp(MathF.Round((values[source] + 1) * 127.5F), 0, 255);
}
FfmpegMedia.EncodeVideo(args[3], new RgbVideo(pixels, 1536, 1024, 1, 24));
Console.WriteLine(
    $"M7C C# tiled decode: pass; latent_mean={latentStats.Mean:G9}; latent_std={latentStats.StandardDeviation:G9}; " +
    $"decoded_mean={decodedStats.Mean:G9}; decoded_std={decodedStats.StandardDeviation:G9}");
return 0;

static (float Mean, float StandardDeviation) Stats(Tensor tensor)
{
    using var resolved = tensor.to(ScalarType.Float32);
    return (resolved.mean().item<float>(), resolved.std().item<float>());
}

static (long Elements, float MaximumAbsoluteError, float Rmse, float Cosine, float ExactFraction) Compare(
    Tensor actual,
    Tensor expected)
{
    using var actualCpu = actual.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
    using var expectedCpu = expected.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
    var left = actualCpu.data<float>().ToArray();
    var right = expectedCpu.data<float>().ToArray();
    if (left.Length != right.Length) throw new InvalidDataException("Transformer oracle shape mismatch.");
    double squared = 0;
    double dot = 0;
    double leftSquared = 0;
    double rightSquared = 0;
    var maximum = 0F;
    long exact = 0;
    for (var index = 0; index < left.Length; index++)
    {
        var difference = left[index] - right[index];
        maximum = MathF.Max(maximum, MathF.Abs(difference));
        squared += difference * difference;
        dot += left[index] * right[index];
        leftSquared += left[index] * left[index];
        rightSquared += right[index] * right[index];
        if (left[index] == right[index]) exact++;
    }
    return (
        left.LongLength,
        maximum,
        (float)Math.Sqrt(squared / left.Length),
        (float)(dot / Math.Sqrt(leftSquared * rightSquared)),
        exact / (float)left.Length);
}

static Tensor VideoPositions(int frames, int height, int width, double fps)
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

static Tensor AudioPositions(int tokens)
{
    var values = new float[checked(tokens * 2)];
    for (var token = 0; token < tokens; token++)
    {
        values[token * 2] = Math.Max(token * 4 + 1 - 4, 0) * 0.01F;
        values[token * 2 + 1] = Math.Max((token + 1) * 4 + 1 - 4, 0) * 0.01F;
    }
    return tensor(values, [1, 1, tokens, 2], dtype: ScalarType.Float32, device: CUDA);
}
