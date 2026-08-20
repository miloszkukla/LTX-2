using System.Diagnostics;
using System.Text.Json;
using Ltx.Core;
using Ltx.SafeTensors;
using Ltx.Trainer;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 4)
{
    Console.Error.WriteLine(
        "usage: Ltx.StandardTrainerTests <checkpoint> <oracle.safetensors> <libLibTorchSharp.so> <report.json>");
    return 2;
}

TorchSharpRuntime.Initialize(args[2]);
InitializeDeviceType(DeviceType.CUDA);
TorchSharpRuntime.UseMathSdpOnly();
var oracle = SafeTensorIndex.Open(Path.GetFullPath(args[1]));
var stopwatch = Stopwatch.StartNew();
using var store = new CheckpointTensorStore([Path.GetFullPath(args[0])]);
using var scope = NewDisposeScope();
var sigma = Load("sigma");
var result = StandardCheckpointLoraTrainer.TrainOneStep(
    store,
    new CheckpointTransformerInput(
        Load("video_latent"), Load("video_context"), null, Load("timesteps"), sigma, Load("video_positions")),
    new CheckpointTransformerInput(
        Load("audio_latent"), Load("audio_context"), null, Load("timesteps"), sigma, Load("audio_positions")),
    Load("video_velocity"),
    Load("audio_velocity"),
    new StandardCheckpointTrainingOptions(
        Rank: 2,
        Alpha: 2,
        LearningRate: 1e-3,
        WeightDecay: 0.01,
        BatchSize: 1,
        MixedPrecision: "bf16",
        BaseQuantization: "none",
        Optimizer: "cuda-adamw"),
    CUDA);
if (result.LayerCount != 48 || result.ModelVersion != "2.3.0" || result.TrainableParameterCount != 8_448 ||
    result.FrozenParameterCount < 20_000_000_000 || result.CudaDeviceCount != 1 ||
    result.MixedPrecision != "bf16" || result.BaseQuantization != "none" ||
    result.OptimizerStateResidence != "cuda_adamw" || result.BatchSize != 1)
{
    throw new InvalidDataException("Standard-profile checkpoint training contract failed.");
}
stopwatch.Stop();
var report = new
{
    schema_version = 1,
    result = "pass",
    execution = "native_csharp_full_checkpoint_backward",
    python_runtime_calls = 0,
    profile = "standard_single_gpu",
    checkpoint_model_version = result.ModelVersion,
    layer_count = result.LayerCount,
    batch_size = result.BatchSize,
    mixed_precision = result.MixedPrecision,
    base_quantization = result.BaseQuantization,
    optimizer = "adamw",
    optimizer_state_residence = result.OptimizerStateResidence,
    trainable_parameter_count = result.TrainableParameterCount,
    frozen_parameter_count = result.FrozenParameterCount,
    loss = result.Loss,
    gradient_norm = result.GradientNorm,
    parameter_delta_norm = result.ParameterDeltaNorm,
    cuda_device_count = result.CudaDeviceCount,
    test_totals = new { passed = 1, failed = 0, skipped = 0 },
    elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
    secrets_recorded = false,
};
var reportPath = Path.GetFullPath(args[3]);
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine(
    $"M7A standard checkpoint trainer: 1 passed, 0 failed; loss={result.Loss:g9}, " +
    $"gradient_norm={result.GradientNorm:g9}, delta_norm={result.ParameterDeltaNorm:g9}");
return 0;

Tensor Load(string name) => oracle.LoadTorchTensor(name, CUDA);
