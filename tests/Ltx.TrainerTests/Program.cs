using System.Text.Json;
using Ltx.Core;
using Ltx.SafeTensors;
using Ltx.Trainer;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: Ltx.TrainerTests <fixture-directory> <libLibTorchSharp.so> <output-directory>");
    return 2;
}

var fixtureDirectory = Path.GetFullPath(args[0]);
var outputDirectory = Path.GetFullPath(args[2]);
using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDirectory, "manifest.json")));
var root = document.RootElement;
if (root.GetProperty("schema_version").GetInt32() != 1 ||
    root.GetProperty("fixture_revision").GetString() != "m6-low-vram-training-v1" ||
    root.GetProperty("oracle").GetProperty("framework_version").GetString() != AbiContract.PyTorchVersion + "+cu132" ||
    root.GetProperty("oracle").GetProperty("cuda_version").GetString() != AbiContract.CudaDisplayVersion ||
    root.GetProperty("oracle").GetProperty("seed").GetInt32() != 20260820)
{
    throw new InvalidDataException("M6 fixture ABI or revision metadata is invalid.");
}

TorchSharpRuntime.Initialize(args[1]);
InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available())
{
    Console.Error.WriteLine("TorchSharp CUDA backend is unavailable.");
    return 3;
}

var fp32 = Tolerance(root.GetProperty("tolerances").GetProperty("fp32"));
var bf16 = Tolerance(root.GetProperty("tolerances").GetProperty("bf16"));
var passed = 0;
var suiteTotals = new Dictionary<string, int>(StringComparer.Ordinal);
var preprocessing = root.GetProperty("preprocessing");
var preprocessingInputs = preprocessing.GetProperty("inputs");
Directory.CreateDirectory(outputDirectory);
TrainingPreprocessResult paths;
using (var scope = NewDisposeScope())
{
    paths = TrainingPreprocessor.Process(
        outputDirectory,
        preprocessing.GetProperty("sample_id").GetString()!,
        new TrainingPreprocessInput(
            Tensor(preprocessingInputs.GetProperty("video"), CPU),
            Tensor(preprocessingInputs.GetProperty("video_encoder_weight"), CPU),
            Tensor(preprocessingInputs.GetProperty("video_encoder_bias"), CPU),
            Tensor(preprocessingInputs.GetProperty("caption_features"), CPU),
            Tensor(preprocessingInputs.GetProperty("connector_weight"), CPU),
            Tensor(preprocessingInputs.GetProperty("connector_bias"), CPU),
            Tensor(preprocessingInputs.GetProperty("prompt_attention_mask"), CPU),
            preprocessing.GetProperty("fps").GetDouble()));
}

var expectedPreprocessing = preprocessing.GetProperty("expected");
var videoFile = SafeTensorFile.Load(paths.VideoLatentsPath);
var conditionsFile = SafeTensorFile.Load(paths.ConditionsPath);
using (var actual = videoFile.Tensors["video_latents"].ToTorchTensor())
{
    AssertTensor("preprocessing video latents", actual, expectedPreprocessing.GetProperty("video_latents"), fp32);
    Pass("preprocessing");
}
using (var actual = conditionsFile.Tensors["video_prompt_embeds"].ToTorchTensor())
{
    AssertTensor("preprocessing prompt embeds", actual, expectedPreprocessing.GetProperty("video_prompt_embeds"), fp32);
    Pass("preprocessing");
}
using (var actual = conditionsFile.Tensors["prompt_attention_mask"].ToTorchTensor())
{
    AssertTensor("preprocessing attention mask", actual, expectedPreprocessing.GetProperty("prompt_attention_mask"), fp32);
    Pass("preprocessing");
}

var dataset = new PrecomputedTrainingDataset(outputDirectory);
using var sample = dataset.Load(0);
if (dataset.Count != 1 || dataset.SampleIds[0] != "sample-0001" || sample.NumFrames != 2 || sample.Height != 2 ||
    sample.Width != 2 || sample.FramesPerSecond != 24 || videoFile.Metadata["schema"] != TrainingPreprocessor.Schema ||
    conditionsFile.Metadata["sample_id"] != sample.SampleId)
{
    throw new InvalidDataException("Precomputed dataset discovery or metadata validation failed.");
}
Pass("preprocessing");
try
{
    using var scope = NewDisposeScope();
    TrainingPreprocessor.Process(
        outputDirectory,
        sample.SampleId,
        new TrainingPreprocessInput(
            Tensor(preprocessingInputs.GetProperty("video"), CPU),
            Tensor(preprocessingInputs.GetProperty("video_encoder_weight"), CPU),
            Tensor(preprocessingInputs.GetProperty("video_encoder_bias"), CPU),
            Tensor(preprocessingInputs.GetProperty("caption_features"), CPU),
            Tensor(preprocessingInputs.GetProperty("connector_weight"), CPU),
            Tensor(preprocessingInputs.GetProperty("connector_bias"), CPU),
            Tensor(preprocessingInputs.GetProperty("prompt_attention_mask"), CPU),
            preprocessing.GetProperty("fps").GetDouble()));
    throw new InvalidDataException("Preprocessing unexpectedly overwrote an existing sample.");
}
catch (IOException)
{
    Pass("preprocessing");
}

var training = root.GetProperty("training");
var baseCase = training.GetProperty("base_weight");
var baseExpected = training.GetProperty("base_quantized");
var baseWeight = RowwiseInt8Matrix.Quantize(
    Floats(baseCase.GetProperty("values")),
    baseExpected.GetProperty("rows").GetInt32(),
    baseExpected.GetProperty("columns").GetInt32());
AssertExact("base INT8 values", baseWeight.Values, SBytes(baseExpected.GetProperty("values")));
AssertNear("base INT8 scales", baseWeight.Scales, Floats(baseExpected.GetProperty("scales")), fp32);
AssertNear("base INT8 dequantized", baseWeight.Dequantize(), Floats(baseExpected.GetProperty("dequantized")), fp32);
Pass("training");

var optionsJson = training.GetProperty("options");
var options = new LowVramTrainingOptions(
    optionsJson.GetProperty("alpha").GetDouble(),
    optionsJson.GetProperty("learning_rate").GetDouble(),
    optionsJson.GetProperty("beta1").GetDouble(),
    optionsJson.GetProperty("beta2").GetDouble(),
    optionsJson.GetProperty("epsilon").GetDouble(),
    optionsJson.GetProperty("weight_decay").GetDouble(),
    optionsJson.GetProperty("max_gradient_norm").GetDouble(),
    optionsJson.GetProperty("activation_chunk_size").GetInt32(),
    optionsJson.GetProperty("gradient_checkpointing").GetBoolean(),
    optionsJson.GetProperty("mixed_precision").GetString()!,
    optionsJson.GetProperty("base_quantization").GetString()!,
    optionsJson.GetProperty("optimizer").GetString()!,
    optionsJson.GetProperty("batch_size").GetInt32());
options.Validate();
Pass("low_vram");

var loraA = training.GetProperty("lora_a");
var loraB = training.GetProperty("lora_b");
var rank = checked((int)loraA.GetProperty("shape")[0].GetInt64());
var inputDimension = checked((int)loraA.GetProperty("shape")[1].GetInt64());
var outputDimension = checked((int)loraB.GetProperty("shape")[0].GetInt64());
using var noise = Tensor(training.GetProperty("noise"), CPU);
var result = LowVramLoraTrainer.TrainOneStep(
    sample,
    noise,
    (float)optionsJson.GetProperty("sigma").GetDouble(),
    baseWeight,
    new LoraParameterSet(
        Floats(loraA.GetProperty("values")),
        Floats(loraB.GetProperty("values")),
        rank,
        inputDimension,
        outputDimension),
    options,
    CUDA);

var expected = training.GetProperty("expected");
AssertNear("noisy latents", result.NoisyLatents, Floats(expected.GetProperty("noisy_latents").GetProperty("values")), fp32);
Pass("training");
AssertNear("velocity targets", result.VelocityTargets, Floats(expected.GetProperty("velocity_targets").GetProperty("values")), fp32);
Pass("training");
AssertNear("loss", [result.Loss], [expected.GetProperty("loss").GetSingle()], bf16);
Pass("training");
AssertNear("LoRA A gradient", result.LoraAGradient, Floats(expected.GetProperty("lora_a_gradient").GetProperty("values")), bf16);
Pass("training");
AssertNear("LoRA B gradient", result.LoraBGradient, Floats(expected.GetProperty("lora_b_gradient").GetProperty("values")), bf16);
Pass("training");
AssertNear(
    "gradient norm and clip scale",
    [result.GradientGlobalNorm, result.GradientClipScale],
    [expected.GetProperty("gradient_global_norm").GetSingle(), expected.GetProperty("gradient_clip_scale").GetSingle()],
    bf16);
Pass("training");
AssertNear("updated LoRA A", result.UpdatedLora.A, Floats(expected.GetProperty("updated_lora_a").GetProperty("values")), bf16);
Pass("training");
AssertNear("updated LoRA B", result.UpdatedLora.B, Floats(expected.GetProperty("updated_lora_b").GetProperty("values")), bf16);
Pass("training");
AssertOptimizerState("LoRA A", result.LoraAOptimizerState, expected.GetProperty("lora_a_optimizer_state"), bf16);
Pass("training");
AssertOptimizerState("LoRA B", result.LoraBOptimizerState, expected.GetProperty("lora_b_optimizer_state"), bf16);
Pass("training");

if (result.BaseWeightResidence != "cpu_int8" || result.OptimizerStateResidence != "cpu_int8")
{
    throw new InvalidDataException("Low-VRAM residency contract failed.");
}
Pass("low_vram");
var fp32OptimizerBytes = 2L * sizeof(float) * result.TrainableParameterCount;
if (result.CpuOptimizerStateBytes >= fp32OptimizerBytes || result.CpuOptimizerStateBytes <= 0)
{
    throw new InvalidDataException("Quantized optimizer state did not reduce storage below FP32 moments.");
}
Pass("low_vram");
if (result.TrainableParameterCount != 16 || result.FrozenParameterCount != 16 ||
    result.UpdatedLora.Rank != rank || result.UpdatedLora.InputDimension != inputDimension ||
    result.UpdatedLora.OutputDimension != outputDimension)
{
    throw new InvalidDataException("LoRA-only trainable/frozen parameter accounting failed.");
}
Pass("low_vram");
if (result.CudaDeviceCount != 1)
{
    throw new InvalidDataException($"M6 requires exactly one visible CUDA device, found {result.CudaDeviceCount}.");
}
Pass("low_vram");

if (passed != 21 || suiteTotals.GetValueOrDefault("preprocessing") != 5 ||
    suiteTotals.GetValueOrDefault("training") != 11 || suiteTotals.GetValueOrDefault("low_vram") != 5)
{
    throw new InvalidDataException("M6 fixture suite totals are incomplete.");
}

Console.WriteLine(
    $"M6 low-VRAM training fixtures: {passed} passed, 0 failed; preprocessing=5, training=11, low_vram=5; " +
    $"loss={result.Loss:g9}, gradient_norm={result.GradientGlobalNorm:g9}, optimizer_state_bytes={result.CpuOptimizerStateBytes}");
return 0;

static Tensor Tensor(JsonElement element, Device device) => tensor(
    Floats(element.GetProperty("values")),
    element.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray(),
    device: device);

static void AssertTensor(string name, Tensor actual, JsonElement expected, NumericTolerance tolerance)
{
    var shape = expected.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
    if (!actual.shape.SequenceEqual(shape))
    {
        throw new InvalidDataException($"{name}: shape mismatch.");
    }
    using var resolved = actual.to(ScalarType.Float32, CPU, non_blocking: false);
    using var contiguous = resolved.contiguous();
    AssertNear(name, contiguous.data<float>().ToArray(), Floats(expected.GetProperty("values")), tolerance);
}

static void AssertOptimizerState(
    string name,
    QuantizedAdamWState actual,
    JsonElement expected,
    NumericTolerance tolerance)
{
    AssertExact(name + " first moment", actual.FirstMoment.Values, SBytes(expected.GetProperty("first_values")));
    AssertNear(name + " first scales", actual.FirstMoment.Scales, Floats(expected.GetProperty("first_scales")), tolerance);
    AssertExact(name + " second moment", actual.SecondMoment.Values, SBytes(expected.GetProperty("second_values")));
    AssertNear(name + " second scales", actual.SecondMoment.Scales, Floats(expected.GetProperty("second_scales")), tolerance);
}

static void AssertNear(
    string name,
    IReadOnlyList<float> actual,
    IReadOnlyList<float> expected,
    NumericTolerance tolerance)
{
    if (actual.Count != expected.Count)
    {
        throw new InvalidDataException($"{name}: length mismatch.");
    }
    for (var index = 0; index < actual.Count; index++)
    {
        var allowed = tolerance.Absolute + tolerance.Relative * MathF.Abs(expected[index]);
        var error = MathF.Abs(actual[index] - expected[index]);
        if (error > allowed)
        {
            throw new InvalidDataException(
                $"{name}[{index}] mismatch: actual={actual[index]:g9}, expected={expected[index]:g9}, allowed={allowed:g9}.");
        }
    }
}

static void AssertExact<T>(string name, IReadOnlyList<T> actual, IReadOnlyList<T> expected)
{
    if (actual.Count != expected.Count || !actual.SequenceEqual(expected))
    {
        throw new InvalidDataException($"{name}: exact sequence mismatch.");
    }
}

static float[] Floats(JsonElement element) => element.EnumerateArray().Select(item => item.GetSingle()).ToArray();

static sbyte[] SBytes(JsonElement element) => element.EnumerateArray().Select(item => checked((sbyte)item.GetInt32())).ToArray();

static NumericTolerance Tolerance(JsonElement element) =>
    new(element.GetProperty("rtol").GetSingle(), element.GetProperty("atol").GetSingle());

void Pass(string suite)
{
    passed++;
    suiteTotals[suite] = suiteTotals.GetValueOrDefault(suite) + 1;
}

readonly record struct NumericTolerance(float Relative, float Absolute);
