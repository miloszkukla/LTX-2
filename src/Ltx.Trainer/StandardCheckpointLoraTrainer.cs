using Ltx.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Ltx.Trainer;

public sealed record StandardCheckpointTrainingOptions(
    int Rank,
    double Alpha,
    double LearningRate,
    double WeightDecay,
    int BatchSize,
    string MixedPrecision,
    string BaseQuantization,
    string Optimizer)
{
    public void Validate()
    {
        if (Rank <= 0 || Alpha <= 0 || LearningRate <= 0 || WeightDecay < 0 || BatchSize != 1 ||
            MixedPrecision != "bf16" || BaseQuantization != "none" || Optimizer != "cuda-adamw")
        {
            throw new InvalidOperationException(
                "The standard profile requires rank-positive LoRA, BF16 unquantized base weights, " +
                "CUDA AdamW, and batch size one.");
        }
    }
}

public sealed record StandardCheckpointTrainingResult(
    float Loss,
    float GradientNorm,
    float ParameterDeltaNorm,
    long TrainableParameterCount,
    long FrozenParameterCount,
    int LayerCount,
    string ModelVersion,
    string MixedPrecision,
    string BaseQuantization,
    string OptimizerStateResidence,
    int BatchSize,
    int CudaDeviceCount);

/// <summary>One-step standard single-GPU LoRA training against the full production checkpoint.</summary>
public static class StandardCheckpointLoraTrainer
{
    private const string Target = "model.diffusion_model.patchify_proj";

    public static StandardCheckpointTrainingResult TrainOneStep(
        CheckpointTensorStore store,
        CheckpointTransformerInput video,
        CheckpointTransformerInput audio,
        Tensor targetVideo,
        Tensor targetAudio,
        StandardCheckpointTrainingOptions options,
        Device device)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(video);
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(targetVideo);
        ArgumentNullException.ThrowIfNull(targetAudio);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(device);
        options.Validate();
        if (!cuda.is_available() || cuda.device_count() != 1)
        {
            throw new InvalidOperationException("Standard checkpoint training requires exactly one CUDA device.");
        }

        using var scope = NewDisposeScope();
        manual_seed(20260820);
        cuda.manual_seed_all(20260820);
        var descriptor = store.Describe(Target + ".weight");
        var outputDimension = descriptor.Shape[0];
        var inputDimension = descriptor.Shape[1];
        using var a = nn.Parameter(
            randn([options.Rank, inputDimension], dtype: ScalarType.BFloat16, device: device).mul(0.05));
        using var b = nn.Parameter(
            randn([outputDimension, options.Rank], dtype: ScalarType.BFloat16, device: device).mul(0.05));
        using var aBefore = a.detach().clone();
        using var bBefore = b.detach().clone();
        var adapter = new CheckpointLoraAdapter(a, b, options.Alpha / options.Rank);
        using var runtime = new CheckpointTransformerRuntime(
            store,
            device,
            new Dictionary<string, CheckpointLoraAdapter>(StringComparer.Ordinal) { [Target] = adapter });
        using var optimizer = optim.AdamW([a, b], lr: options.LearningRate, weight_decay: options.WeightDecay);
        optimizer.zero_grad();
        using var output = runtime.Forward(video, audio);
        using var videoLoss = output.Video!.to(ScalarType.Float32).sub(targetVideo.to(output.Video.device))
            .pow(2).mean();
        using var audioLoss = output.Audio!.to(ScalarType.Float32).sub(targetAudio.to(output.Audio.device))
            .pow(2).mean();
        using var loss = videoLoss.add(audioLoss);
        loss.backward();
        var aGradient = a.grad ?? throw new InvalidOperationException("Standard LoRA A gradient is missing.");
        var bGradient = b.grad ?? throw new InvalidOperationException("Standard LoRA B gradient is missing.");
        using var gradientNormTensor = aGradient.to(ScalarType.Float32).pow(2).sum()
            .add(bGradient.to(ScalarType.Float32).pow(2).sum()).sqrt();
        var gradientNorm = Scalar(gradientNormTensor);
        if (!float.IsFinite(gradientNorm) || gradientNorm <= 0)
        {
            throw new InvalidDataException("Standard checkpoint training produced an invalid LoRA gradient.");
        }
        optimizer.step();
        using var parameterDelta = a.to(ScalarType.Float32).sub(aBefore.to(ScalarType.Float32)).pow(2).sum()
            .add(b.to(ScalarType.Float32).sub(bBefore.to(ScalarType.Float32)).pow(2).sum()).sqrt();
        var deltaNorm = Scalar(parameterDelta);
        if (!float.IsFinite(deltaNorm) || deltaNorm <= 0)
        {
            throw new InvalidDataException("CUDA AdamW did not update the standard-profile LoRA parameters.");
        }
        var frozenParameters = store.TensorNames
            .Where(name => name.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
            .Sum(name => store.Describe(name).Shape.Aggregate(1L, (count, dimension) => checked(count * dimension)));
        return new StandardCheckpointTrainingResult(
            Scalar(loss),
            gradientNorm,
            deltaNorm,
            checked((long)a.numel() + b.numel()),
            frozenParameters,
            runtime.LayerCount,
            runtime.ModelVersion,
            options.MixedPrecision,
            options.BaseQuantization,
            "cuda_adamw",
            options.BatchSize,
            checked((int)cuda.device_count()));
    }

    private static float Scalar(Tensor value)
    {
        using var resolved = value.detach().to(ScalarType.Float32, CPU, non_blocking: false).contiguous();
        return resolved.data<float>()[0];
    }
}
