using System.Diagnostics;
using System.Text.Json;
using Ltx.Core;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 4)
{
    Console.Error.WriteLine(
        "usage: Ltx.CheckpointRuntimeTests <checkpoint> <oracle.safetensors> <libLibTorchSharp.so> <result.json>");
    return 2;
}

var checkpointPath = Path.GetFullPath(args[0]);
var oraclePath = Path.GetFullPath(args[1]);
var resultPath = Path.GetFullPath(args[3]);
TorchSharpRuntime.Initialize(args[2]);
InitializeDeviceType(DeviceType.CUDA);
TorchSharpRuntime.UseMathSdpOnly();
if (!cuda.is_available() || cuda.device_count() != 1)
{
    throw new InvalidOperationException("M7A checkpoint tests require exactly one CUDA device.");
}

var stopwatch = Stopwatch.StartNew();
var oracle = SafeTensorIndex.Open(oraclePath);
if (oracle.Metadata.GetValueOrDefault("fixture_revision") != "m7a-real-checkpoint-runtime-v4" ||
    oracle.Metadata.GetValueOrDefault("attention_backend") != "pytorch_sdpa_math")
{
    throw new InvalidDataException("M7A oracle revision mismatch.");
}
var passed = 0;
var comparisons = new Dictionary<string, object>(StringComparer.Ordinal);

using var store = new CheckpointTensorStore([checkpointPath]);
if (store.Indexes.Count != 1 || store.Indexes[0].Tensors.Count != 5947 ||
    store.Metadata.GetValueOrDefault("model_version") != "2.3.0")
{
    throw new InvalidDataException("Pinned LTX-2.3 monolith metadata/tensor inventory mismatch.");
}
passed++;

using (var scope = NewDisposeScope())
using (var runtime = new CheckpointTransformerRuntime(store, CUDA))
{
    var video = new CheckpointTransformerInput(
        Load("video_latent"),
        Load("video_context"),
        null,
        Load("timesteps"),
        Load("sigma"),
        Load("video_positions"));
    var audio = new CheckpointTransformerInput(
        Load("audio_latent"),
        Load("audio_context"),
        null,
        Load("timesteps"),
        Load("sigma"),
        Load("audio_positions"));
    using var actual = runtime.Forward(
        video,
        audio,
        observeBlock: (layer, videoState, audioState) =>
        {
            var suffix = layer < 0 ? "prepared" : $"block_{layer}";
            Compare($"transformer_video_{suffix}", videoState!, Load($"transformer_video_{suffix}"), 0, 0);
            Compare($"transformer_audio_{suffix}", audioState!, Load($"transformer_audio_{suffix}"), 0, 0);
        },
        observeTensor: (prefix, tensor) =>
        {
            if (prefix.StartsWith("video_output_", StringComparison.Ordinal))
            {
                Compare($"transformer_{prefix}", tensor, Load($"transformer_{prefix}"), 0, 0);
                return;
            }
            const string blockPrefix = "model.diffusion_model.transformer_blocks.0.";
            if (!prefix.StartsWith(blockPrefix, StringComparison.Ordinal)) return;
            var oracleName = $"transformer_block_0_{prefix[blockPrefix.Length..]}";
            if (oracle.Tensors.ContainsKey(oracleName)) Compare(oracleName, tensor, Load(oracleName), 0, 0);
        });
    Compare("transformer_video_velocity", actual.Video!, Load("video_velocity"));
    Compare("transformer_audio_velocity", actual.Audio!, Load("audio_velocity"));
    if (runtime.LayerCount != 48 || runtime.ModelVersion != "2.3.0")
    {
        throw new InvalidDataException("Transformer did not auto-detect the production model generation.");
    }
    passed++;
}
store.ClearCache();

using (var scope = NewDisposeScope())
{
    var decoder = new CheckpointVideoDecoder(store, CUDA);
    using var actual = decoder.Decode(Load("video_decoder_latent"));
    Compare("convolutional_video_decoder", actual, Load("decoded_video"));
    passed++;
}
store.ClearCache();

using (var scope = NewDisposeScope())
{
    var decoder = new CheckpointAudioDecoder(store, CUDA);
    using var actual = decoder.Decode(Load("audio_decoder_latent"));
    Compare("audio_vae_decoder", actual, Load("decoded_audio"));
    if (decoder.SampleRate != 16000 || decoder.MelBins != 64)
    {
        throw new InvalidDataException("Audio decoder metadata auto-detection mismatch.");
    }
    passed++;
}
store.ClearCache();

using (var scope = NewDisposeScope())
{
    var vocoder = new CheckpointVocoder(store, CUDA);
    using var actual = vocoder.Decode(Load("decoded_audio"));
    Compare("vocoder_with_bandwidth_extension", actual, Load("decoded_waveform"));
    if (vocoder.OutputSampleRate != 48_000)
    {
        throw new InvalidDataException("Vocoder output sample-rate metadata auto-detection mismatch.");
    }
    passed++;
}

stopwatch.Stop();
var result = new
{
    schema_version = 1,
    result = "pass",
    fixture_revision = oracle.Metadata["fixture_revision"],
    checkpoint_model_version = store.Metadata["model_version"],
    checkpoint_tensor_count = store.Indexes[0].Tensors.Count,
    execution = "native_csharp_torchsharp_cuda_full_checkpoint",
    python_runtime_calls = 0,
    test_totals = new { passed, failed = 0, skipped = 0 },
    numerical_tolerances = new
    {
        bf16 = new { rtol = 2e-2, atol = 5e-3, result = "pass" },
    },
    comparisons,
    elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
    secrets_recorded = false,
};
Directory.CreateDirectory(Path.GetDirectoryName(resultPath)!);
File.WriteAllText(resultPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"M7A real checkpoint runtime: {passed} passed, 0 failed in {stopwatch.Elapsed.TotalSeconds:F1}s");
return 0;

Tensor Load(string name) => oracle.LoadTorchTensor(name, CUDA);

void Compare(string name, Tensor actual, Tensor expected, float absoluteTolerance = 0.005F, float relativeTolerance = 0.02F)
{
    if (!actual.shape.SequenceEqual(expected.shape))
    {
        throw new InvalidDataException($"{name}: shape mismatch, actual=[{string.Join(',', actual.shape)}], " +
            $"expected=[{string.Join(',', expected.shape)}].");
    }
    using var actualFloat = actual.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
    using var expectedFloat = expected.to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
    var actualValues = actualFloat.data<float>().ToArray();
    var expectedValues = expectedFloat.data<float>().ToArray();
    var maximumAbsolute = 0F;
    var maximumRelative = 0F;
    for (var index = 0; index < actualValues.Length; index++)
    {
        var absolute = MathF.Abs(actualValues[index] - expectedValues[index]);
        var relative = absolute / MathF.Max(MathF.Abs(expectedValues[index]), 1e-12F);
        maximumAbsolute = MathF.Max(maximumAbsolute, absolute);
        maximumRelative = MathF.Max(maximumRelative, relative);
        if (!float.IsFinite(actualValues[index]) ||
            absolute > absoluteTolerance + relativeTolerance * MathF.Abs(expectedValues[index]))
        {
            throw new InvalidDataException(
                $"{name}[{index}]: actual={actualValues[index]:g9}, expected={expectedValues[index]:g9}, " +
                $"absolute={absolute:g9}, relative={relative:g9}.");
        }
    }
    comparisons[name] = new
    {
        elements = actualValues.Length,
        max_absolute_error = maximumAbsolute,
        max_relative_error = maximumRelative,
        result = "pass",
    };
}
