using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Trainer;

public sealed record RowwiseInt8Matrix(sbyte[] Values, float[] Scales, int Rows, int Columns)
{
    public static RowwiseInt8Matrix Quantize(IReadOnlyList<float> source, int rows, int columns)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (rows <= 0 || columns <= 0 || source.Count != checked(rows * columns))
        {
            throw new ArgumentException("Rowwise INT8 source shape is invalid.", nameof(source));
        }
        var values = new sbyte[source.Count];
        var scales = new float[rows];
        for (var row = 0; row < rows; row++)
        {
            var maximum = 0F;
            for (var column = 0; column < columns; column++)
            {
                maximum = MathF.Max(maximum, MathF.Abs(source[row * columns + column]));
            }
            var scale = maximum > 0 ? maximum / 127F : 0F;
            scales[row] = scale;
            for (var column = 0; column < columns; column++)
            {
                var normalized = scale > 0 ? source[row * columns + column] / scale : 0F;
                var rounded = MathF.Round(Math.Clamp(normalized, -127F, 127F), MidpointRounding.ToEven);
                values[row * columns + column] = checked((sbyte)rounded);
            }
        }
        return new RowwiseInt8Matrix(values, scales, rows, columns);
    }

    public float[] Dequantize()
    {
        if (Rows <= 0 || Columns <= 0 || Values.Length != checked(Rows * Columns) || Scales.Length != Rows)
        {
            throw new InvalidDataException("Rowwise INT8 matrix payload is invalid.");
        }
        var result = new float[Values.Length];
        for (var row = 0; row < Rows; row++)
        {
            for (var column = 0; column < Columns; column++)
            {
                result[row * Columns + column] = Values[row * Columns + column] * Scales[row];
            }
        }
        return result;
    }

    public long StorageBytes => Values.LongLength + sizeof(float) * Scales.LongLength;
}

public sealed record LoraParameterSet(
    float[] A,
    float[] B,
    int Rank,
    int InputDimension,
    int OutputDimension);

public sealed record LowVramTrainingOptions(
    double Alpha,
    double LearningRate,
    double Beta1,
    double Beta2,
    double Epsilon,
    double WeightDecay,
    double MaxGradientNorm,
    int ActivationChunkSize,
    bool GradientCheckpointing,
    string MixedPrecision,
    string BaseQuantization,
    string Optimizer,
    int BatchSize)
{
    public void Validate()
    {
        if (Alpha <= 0 || LearningRate <= 0 || Beta1 is < 0 or >= 1 || Beta2 is < 0 or >= 1 ||
            Epsilon <= 0 || WeightDecay < 0 || MaxGradientNorm <= 0 || ActivationChunkSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(LowVramTrainingOptions), "Low-VRAM optimizer settings are invalid.");
        }
        if (!GradientCheckpointing || MixedPrecision != "bf16" || BaseQuantization != "int8-rowwise" ||
            Optimizer != "cpu-offloaded-adamw8bit" || BatchSize != 1)
        {
            throw new InvalidOperationException(
                "The M6 low-VRAM profile requires BF16, rowwise INT8 base weights, CPU-offloaded AdamW8bit, " +
                "gradient checkpointing, and batch size one.");
        }
    }
}

public sealed record QuantizedAdamWState(RowwiseInt8Matrix FirstMoment, RowwiseInt8Matrix SecondMoment)
{
    public long StorageBytes => FirstMoment.StorageBytes + SecondMoment.StorageBytes;
}

public sealed record OneStepTrainingResult(
    float Loss,
    float[] NoisyLatents,
    float[] VelocityTargets,
    float[] LoraAGradient,
    float[] LoraBGradient,
    float GradientGlobalNorm,
    float GradientClipScale,
    LoraParameterSet UpdatedLora,
    QuantizedAdamWState LoraAOptimizerState,
    QuantizedAdamWState LoraBOptimizerState,
    long TrainableParameterCount,
    long FrozenParameterCount,
    long CpuOptimizerStateBytes,
    string BaseWeightResidence,
    string OptimizerStateResidence,
    int CudaDeviceCount);

public static class LowVramLoraTrainer
{
    public static OneStepTrainingResult TrainOneStep(
        PrecomputedTrainingSample sample,
        Tensor noise,
        float sigma,
        RowwiseInt8Matrix baseWeight,
        LoraParameterSet lora,
        LowVramTrainingOptions options,
        Device device)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(baseWeight);
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(device);
        options.Validate();
        Validate(sample, noise, sigma, baseWeight, lora);
        if (!cuda.is_available())
        {
            throw new InvalidOperationException("M6 one-step LoRA training requires a CUDA device.");
        }

        using var scope = NewDisposeScope();
        var latentChannels = sample.VideoLatents.shape[0];
        var tokenCount = checked((int)(sample.VideoLatents.shape[1] * sample.VideoLatents.shape[2] *
            sample.VideoLatents.shape[3]));
        var latents = sample.VideoLatents.permute(1, 2, 3, 0)
            .reshape(1, tokenCount, latentChannels)
            .to(ScalarType.Float32, device, non_blocking: false);
        var prompt = sample.VideoPromptEmbeds.to(ScalarType.Float32, device, non_blocking: false);
        var mask = sample.PromptAttentionMask.to(ScalarType.Float32, device, non_blocking: false);
        var resolvedNoise = noise.to(ScalarType.Float32, device, non_blocking: false);
        var context = prompt.mul(mask.unsqueeze(1)).sum([0L])
            .div(mask.sum().clamp(min: 1))
            .reshape(1, 1, latentChannels);
        var noisyLatents = latents.mul(1 - sigma).add(resolvedNoise.mul(sigma));
        var targets = resolvedNoise.sub(latents);
        var modelInput = noisyLatents.add(context).to(ScalarType.BFloat16);

        var dequantizedBase = baseWeight.Dequantize();
        var baseTensor = tensor(dequantizedBase, [baseWeight.Rows, baseWeight.Columns], device: device)
            .to(ScalarType.BFloat16, disposeAfter: true);
        var loraA = tensor(lora.A, [lora.Rank, lora.InputDimension], device: device)
            .to(ScalarType.BFloat16, disposeAfter: true)
            .requires_grad_(true);
        var loraB = tensor(lora.B, [lora.OutputDimension, lora.Rank], device: device)
            .to(ScalarType.BFloat16, disposeAfter: true)
            .requires_grad_(true);
        var scale = options.Alpha / lora.Rank;
        var totalLoss = 0F;
        for (var start = 0; start < tokenCount; start += options.ActivationChunkSize)
        {
            var length = Math.Min(options.ActivationChunkSize, tokenCount - start);
            var effectiveWeight = baseTensor.add(loraB.matmul(loraA).mul(scale));
            var prediction = modelInput.narrow(1, start, length).matmul(effectiveWeight.transpose(0, 1));
            var chunkTarget = targets.narrow(1, start, length);
            var chunkLoss = prediction.to(ScalarType.Float32).sub(chunkTarget).pow(2).mean();
            var weight = (double)length / tokenCount;
            chunkLoss.mul(weight).backward();
            totalLoss += ScalarValue(chunkLoss) * (float)weight;
        }

        var rawAGradient = FloatValues(loraA.grad ?? throw new InvalidOperationException("LoRA A gradient is missing."));
        var rawBGradient = FloatValues(loraB.grad ?? throw new InvalidOperationException("LoRA B gradient is missing."));
        var globalNorm = MathF.Sqrt(rawAGradient.Sum(value => value * value) + rawBGradient.Sum(value => value * value));
        var clipScale = MathF.Min(1F, (float)(options.MaxGradientNorm / (globalNorm + 1e-6F)));
        var clippedA = rawAGradient.Select(value => value * clipScale).ToArray();
        var clippedB = rawBGradient.Select(value => value * clipScale).ToArray();
        var aBefore = FloatValues(loraA);
        var bBefore = FloatValues(loraB);
        var (updatedA, stateA) = AdamW8BitStep(aBefore, clippedA, lora.Rank, lora.InputDimension, options);
        var (updatedB, stateB) = AdamW8BitStep(bBefore, clippedB, lora.OutputDimension, lora.Rank, options);
        var updated = new LoraParameterSet(updatedA, updatedB, lora.Rank, lora.InputDimension, lora.OutputDimension);
        var optimizerBytes = stateA.StorageBytes + stateB.StorageBytes;
        return new OneStepTrainingResult(
            totalLoss,
            FloatValues(noisyLatents),
            FloatValues(targets),
            rawAGradient,
            rawBGradient,
            globalNorm,
            clipScale,
            updated,
            stateA,
            stateB,
            checked((long)lora.A.Length + lora.B.Length),
            baseWeight.Values.LongLength,
            optimizerBytes,
            "cpu_int8",
            "cpu_int8",
            checked((int)cuda.device_count()));
    }

    private static (float[] Updated, QuantizedAdamWState State) AdamW8BitStep(
        IReadOnlyList<float> parameter,
        IReadOnlyList<float> gradient,
        int rows,
        int columns,
        LowVramTrainingOptions options)
    {
        var first = new float[gradient.Count];
        var second = new float[gradient.Count];
        for (var index = 0; index < gradient.Count; index++)
        {
            first[index] = (float)((1 - options.Beta1) * gradient[index]);
            second[index] = (float)((1 - options.Beta2) * gradient[index] * gradient[index]);
        }
        var firstQuantized = RowwiseInt8Matrix.Quantize(first, rows, columns);
        var secondQuantized = RowwiseInt8Matrix.Quantize(second, rows, columns);
        var firstResolved = firstQuantized.Dequantize();
        var secondResolved = secondQuantized.Dequantize();
        var updated = new float[parameter.Count];
        for (var index = 0; index < updated.Length; index++)
        {
            var firstHat = firstResolved[index] / (1 - options.Beta1);
            var secondHat = secondResolved[index] / (1 - options.Beta2);
            updated[index] = (float)(
                parameter[index] * (1 - options.LearningRate * options.WeightDecay) -
                options.LearningRate * firstHat / (Math.Sqrt(secondHat) + options.Epsilon));
        }
        return (updated, new QuantizedAdamWState(firstQuantized, secondQuantized));
    }

    private static void Validate(
        PrecomputedTrainingSample sample,
        Tensor noise,
        float sigma,
        RowwiseInt8Matrix baseWeight,
        LoraParameterSet lora)
    {
        if (!float.IsFinite(sigma) || sigma is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sigma));
        }
        if (sample.VideoLatents.dim() != 4 || sample.VideoPromptEmbeds.dim() != 2 ||
            sample.PromptAttentionMask.dim() != 1)
        {
            throw new InvalidDataException("Precomputed training tensors have invalid ranks.");
        }
        var channels = checked((int)sample.VideoLatents.shape[0]);
        var tokens = checked((int)(sample.VideoLatents.shape[1] * sample.VideoLatents.shape[2] * sample.VideoLatents.shape[3]));
        if (sample.VideoPromptEmbeds.shape[1] != channels ||
            sample.VideoPromptEmbeds.shape[0] != sample.PromptAttentionMask.shape[0] ||
            !noise.shape.SequenceEqual(new long[] { 1, tokens, channels }))
        {
            throw new InvalidDataException("Precomputed training sample and noise shapes are incompatible.");
        }
        if (baseWeight.Rows != channels || baseWeight.Columns != channels ||
            lora.InputDimension != channels || lora.OutputDimension != channels || lora.Rank <= 0 ||
            lora.A.Length != checked(lora.Rank * channels) || lora.B.Length != checked(channels * lora.Rank))
        {
            throw new InvalidDataException("Base weight and LoRA dimensions are incompatible with the sample.");
        }
    }

    private static float ScalarValue(Tensor tensor)
    {
        using var detached = tensor.detach();
        using var resolved = detached.to(ScalarType.Float32, CPU, non_blocking: false);
        return resolved.data<float>()[0];
    }

    private static float[] FloatValues(Tensor tensor)
    {
        using var detached = tensor.detach();
        using var resolved = detached.to(ScalarType.Float32, CPU, non_blocking: false);
        using var contiguous = resolved.contiguous();
        return contiguous.data<float>().ToArray();
    }
}
