using System.Text.Json;
using Ltx.Core;
using Ltx.CoreExecutionTests;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: Ltx.CoreExecutionTests <fixture-directory> <libLibTorchSharp.so> <offload.safetensors>");
    return 2;
}

var fixtureDirectory = Path.GetFullPath(args[0]);
var fixture = JsonSerializer.Deserialize<FixtureManifest>(File.ReadAllText(Path.Combine(fixtureDirectory, "manifest.json")))
    ?? throw new InvalidDataException("M3 fixture manifest is empty.");
if (fixture.SchemaVersion != 1 || fixture.FixtureRevision != "m3-core-model-execution-v1" ||
    fixture.Oracle.FrameworkVersion != "2.13.0+cu132" || fixture.Oracle.CudaVersion != AbiContract.CudaDisplayVersion ||
    fixture.Oracle.SafetensorsVersion != "0.6.2" || fixture.Oracle.Seed != 20260820)
{
    throw new InvalidDataException("M3 fixture ABI or revision metadata is invalid.");
}

TorchSharpRuntime.Initialize(args[1]);
InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available())
{
    Console.Error.WriteLine("TorchSharp CUDA backend is unavailable.");
    return 3;
}

var passed = 0;
var suiteTotals = new Dictionary<string, int>(StringComparer.Ordinal);
foreach (var (name, fixtureCase) in fixture.Cases.OrderBy(pair => pair.Key, StringComparer.Ordinal))
{
    if (name == "offload.linear")
    {
        foreach (var (precision, expected) in fixtureCase.Expected.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            var dtype = PrecisionType(precision);
            foreach (var mode in new[] { OffloadMode.None, OffloadMode.Cpu, OffloadMode.Disk })
            {
                using var scope = NewDisposeScope();
                using var input = CreateTensor(fixtureCase.Tensors["input"], dtype, CUDA);
                using var sourceWeight = CreateTensor(fixtureCase.Tensors["weight"], ScalarType.Float32, CPU);
                using var sourceBias = CreateTensor(fixtureCase.Tensors["bias"], ScalarType.Float32, CPU);
                using var linear = mode == OffloadMode.Disk
                    ? new OffloadedLinear(args[2], "linear.weight", "linear.bias", CUDA, dtype)
                    : new OffloadedLinear(sourceWeight, sourceBias, mode, CUDA, dtype);
                using var output = linear.Forward(input);
                AssertClose($"{name}.{mode}.{precision}", output, expected, fixture.Tolerances[precision]);
                var expectedResidency = mode switch
                {
                    OffloadMode.None => "cuda",
                    OffloadMode.Cpu => "cpu",
                    OffloadMode.Disk => "disk",
                    _ => throw new InvalidOperationException(),
                };
                if (linear.Residency != expectedResidency || (mode == OffloadMode.Disk && linear.DiskLoads != 1))
                {
                    throw new InvalidDataException($"{name}.{mode}: offload residency or load count mismatch.");
                }
                RecordPass(fixtureCase.Suite);
            }
        }
        continue;
    }

    foreach (var (precision, expected) in fixtureCase.Expected.OrderBy(pair => pair.Key, StringComparer.Ordinal))
    {
        using var scope = NewDisposeScope();
        var dtype = PrecisionType(precision);
        using var output = Execute(name, fixtureCase, dtype);
        AssertClose($"{name}.{precision}", output, expected, fixture.Tolerances[precision]);
        RecordPass(fixtureCase.Suite);
    }
}

if (suiteTotals.Keys.OrderBy(value => value, StringComparer.Ordinal).SequenceEqual(
    new[] { "audio_vocoder", "conditioning", "offload", "transformer", "vae" }) is false)
{
    throw new InvalidDataException("M3 fixture suites are incomplete.");
}

Console.WriteLine(
    $"M3 core fixtures: {passed} passed, 0 failed; " +
    string.Join(", ", suiteTotals.OrderBy(pair => pair.Key).Select(pair => $"{pair.Key}={pair.Value}")) +
    $"; revision={fixture.FixtureRevision}");
return 0;

Tensor Execute(string name, FixtureCase fixtureCase, ScalarType dtype)
{
    Tensor Input(string key) => CreateTensor(fixtureCase.Tensors[key], dtype, CUDA);

    return name switch
    {
        "transformer.timestep_embedding" => TransformerExecution.TimestepEmbedding(
            Input("timesteps"),
            checked((int)fixtureCase.Integers["embedding_dimension"]),
            fixtureCase.Flags["flip_sin_to_cos"],
            fixtureCase.Numbers["downscale_frequency_shift"],
            fixtureCase.Numbers["scale"],
            checked((int)fixtureCase.Integers["maximum_period"])),
        "transformer.feed_forward" => TransformerExecution.FeedForward(
            Input("input"),
            Input("input_weight"),
            Input("input_bias"),
            Input("output_weight"),
            Input("output_bias")),
        "transformer.attention" => TransformerExecution.ScaledDotProductAttention(
            Input("query"), Input("key"), Input("value"), checked((int)fixtureCase.Integers["heads"])),
        "transformer.masked_attention" => TransformerExecution.ScaledDotProductAttention(
            Input("query"),
            Input("key"),
            Input("value"),
            checked((int)fixtureCase.Integers["heads"]),
            Input("mask")),
        "vae.patchify" => VaeExecution.Patchify(
            Input("input"),
            checked((int)fixtureCase.Integers["spatial_patch"]),
            checked((int)fixtureCase.Integers["temporal_patch"])),
        "vae.unpatchify" => VaeExecution.Unpatchify(
            Input("input"),
            checked((int)fixtureCase.Integers["spatial_patch"]),
            checked((int)fixtureCase.Integers["temporal_patch"])),
        "vae.pixel_norm" => VaeExecution.PixelNorm(Input("input")),
        "vae.statistics_normalize" => VaeExecution.NormalizeChannels(
            Input("input"), Input("means"), Input("standard_deviations"), fixtureCase.Integers["dimension"]),
        "vae.statistics_denormalize" => VaeExecution.DenormalizeChannels(
            Input("input"), Input("means"), Input("standard_deviations"), fixtureCase.Integers["dimension"]),
        "audio.snake" => AudioExecution.Snake(
            Input("input"), Input("alpha"), fixtureCase.Flags["logscale"], 1e-9),
        "audio.snake_beta" => AudioExecution.SnakeBeta(
            Input("input"), Input("alpha"), Input("beta"), fixtureCase.Flags["logscale"], 1e-9),
        "vocoder.prepare_stereo_mel" => AudioExecution.PrepareStereoMel(Input("input")),
        "vocoder.final_tanh" or "vocoder.final_clamp" => AudioExecution.ApplyVocoderFinalActivation(
            Input("input"), fixtureCase.Flags["use_tanh"]),
        "conditioning.first_mask" => ConditioningExecution.BuildAttentionMask(
            null,
            fixtureCase.Integers["noisy_tokens"],
            fixtureCase.Integers["new_tokens"],
            fixtureCase.Integers["existing_tokens"],
            Input("cross_mask")),
        "conditioning.append_mask" => ConditioningExecution.BuildAttentionMask(
            Input("existing_mask"),
            fixtureCase.Integers["noisy_tokens"],
            fixtureCase.Integers["new_tokens"],
            fixtureCase.Integers["existing_tokens"],
            Input("cross_mask")),
        "conditioning.resolve_scalar" or "conditioning.resolve_vector" => ConditioningExecution.ResolveCrossMask(
            Input("attention_mask"),
            fixtureCase.Integers["new_tokens"],
            fixtureCase.Integers["batch_size"],
            CUDA,
            dtype),
        _ => throw new InvalidDataException($"Unknown M3 fixture case '{name}'."),
    };
}

void RecordPass(string suite)
{
    passed++;
    suiteTotals[suite] = suiteTotals.GetValueOrDefault(suite) + 1;
}

static ScalarType PrecisionType(string precision) => precision switch
{
    "fp32" => ScalarType.Float32,
    "bf16" => ScalarType.BFloat16,
    _ => throw new InvalidDataException($"Unknown fixture precision '{precision}'."),
};

static Tensor CreateTensor(TensorData data, ScalarType dtype, Device device)
{
    var input = tensor(data.Values, data.Shape, device: device);
    return dtype == ScalarType.Float32 ? input : input.to(dtype, disposeAfter: true);
}

static void AssertClose(string name, Tensor actual, TensorData expected, Tolerance tolerance)
{
    if (!actual.shape.SequenceEqual(expected.Shape))
    {
        throw new InvalidDataException(
            $"{name}: shape [{string.Join(',', actual.shape)}] does not match [{string.Join(',', expected.Shape)}].");
    }

    using var floatOutput = actual.to(ScalarType.Float32);
    using var cpuOutput = floatOutput.cpu();
    var values = cpuOutput.data<float>().ToArray();
    for (var index = 0; index < values.Length; index++)
    {
        var allowed = tolerance.Absolute + tolerance.Relative * MathF.Abs(expected.Values[index]);
        var error = MathF.Abs(values[index] - expected.Values[index]);
        if (error > allowed)
        {
            throw new InvalidDataException(
                $"{name}[{index}] mismatch: actual={values[index]:g9}, expected={expected.Values[index]:g9}, allowed={allowed:g9}.");
        }
    }
}
