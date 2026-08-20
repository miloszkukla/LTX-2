using System.Runtime.InteropServices;

namespace Ltx.Cuda;

public enum Fp8GemmVariant
{
    Sm89 = 0,
    Sm90 = 1,
    Sm90Bias = 2,
}

public sealed record NvFp4Tensor(
    byte[] Packed,
    byte[] BlockScales,
    int Rows,
    int Columns,
    float PerTensorScale,
    bool HighNibbleFirst)
{
    public int PaddedScaleRows => ((Rows + 127) / 128) * 128;

    public int PaddedScaleColumns => (((Columns / 16) + 3) / 4) * 4;

    public NvFp4Tensor DeepClone() => new(
        (byte[])Packed.Clone(),
        (byte[])BlockScales.Clone(),
        Rows,
        Columns,
        PerTensorScale,
        HighNibbleFirst);
}

public sealed record BlockwiseFp8Tensor(
    byte[] Values,
    float[] Scales,
    int Rows,
    int Columns,
    int BlockSize);

public sealed record RowwiseInt8Tensor(
    sbyte[] Values,
    float[] Scales,
    int Rows,
    int Columns);

public sealed record QuantizedLoraFusion(NvFp4Tensor Original, NvFp4Tensor Fused)
{
    public NvFp4Tensor Unfuse() => Original.DeepClone();
}

public static class LtxCudaKernels
{
    public static IReadOnlyList<string> KernelInventory
    {
        get
        {
            var count = LtxCudaNative.KernelCount();
            if (count < 0 || count > 1024)
            {
                throw new InvalidOperationException($"Invalid native kernel count {count}.");
            }

            var result = new string[count];
            for (var index = 0; index < count; index++)
            {
                var pointer = LtxCudaNative.KernelName(index);
                result[index] = Marshal.PtrToStringUTF8(pointer)
                    ?? throw new InvalidOperationException($"Native kernel name {index} is null.");
            }

            return result;
        }
    }

    public static unsafe float[] FusedAddRound(
        ReadOnlySpan<float> delta,
        ReadOnlySpan<float> weight,
        uint seed)
    {
        RequireSameLength(delta, weight, nameof(weight));
        var output = new float[delta.Length];
        if (output.Length == 0) return output;
        fixed (float* deltaPointer = delta)
        fixed (float* weightPointer = weight)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.FusedAddRoundHost(
                deltaPointer, weightPointer, outputPointer, (nuint)output.Length, seed), nameof(FusedAddRound));
        }
        return output;
    }

    public static unsafe float[] NeighborhoodAttention3D(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        int batch,
        int time,
        int height,
        int width,
        int heads,
        int headDimension,
        int kernelTime,
        int kernelHeight,
        int kernelWidth,
        bool causalTime = false,
        bool causalHeight = false,
        bool causalWidth = false,
        float? scale = null)
    {
        var count = checked(batch * time * height * width * heads * headDimension);
        RequireLength(query, count, nameof(query));
        RequireLength(key, count, nameof(key));
        RequireLength(value, count, nameof(value));
        var output = new float[count];
        fixed (float* queryPointer = query)
        fixed (float* keyPointer = key)
        fixed (float* valuePointer = value)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.Na3dHost(
                queryPointer, keyPointer, valuePointer, outputPointer,
                batch, time, height, width, heads, headDimension,
                kernelTime, kernelHeight, kernelWidth,
                causalTime ? 1 : 0, causalHeight ? 1 : 0, causalWidth ? 1 : 0,
                scale ?? 1F / MathF.Sqrt(headDimension)), nameof(NeighborhoodAttention3D));
        }
        return output;
    }

    public static unsafe float[] SwiGluUpMultiply(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> upWeight,
        ReadOnlySpan<float> gate,
        int rows,
        int inputDimension,
        int outputDimension)
    {
        RequireMatrix(input, rows, inputDimension, nameof(input));
        RequireMatrix(upWeight, outputDimension, inputDimension, nameof(upWeight));
        RequireMatrix(gate, rows, outputDimension, nameof(gate));
        var output = new float[checked(rows * outputDimension)];
        fixed (float* inputPointer = input)
        fixed (float* weightPointer = upWeight)
        fixed (float* gatePointer = gate)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.SwiGluUpMultiplyHost(
                inputPointer, weightPointer, gatePointer, outputPointer,
                rows, inputDimension, outputDimension), nameof(SwiGluUpMultiply));
        }
        return output;
    }

    public static unsafe float[] SwiGluGateUp(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gateWeight,
        ReadOnlySpan<float> upWeight,
        int rows,
        int inputDimension,
        int outputDimension)
    {
        RequireMatrix(input, rows, inputDimension, nameof(input));
        RequireMatrix(gateWeight, outputDimension, inputDimension, nameof(gateWeight));
        RequireMatrix(upWeight, outputDimension, inputDimension, nameof(upWeight));
        var output = new float[checked(rows * outputDimension)];
        fixed (float* inputPointer = input)
        fixed (float* gatePointer = gateWeight)
        fixed (float* upPointer = upWeight)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.SwiGluGateUpHost(
                inputPointer, gatePointer, upPointer, outputPointer,
                rows, inputDimension, outputDimension), nameof(SwiGluGateUp));
        }
        return output;
    }

    public static unsafe float[] Fp8Gemm(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        int rows,
        int outputDimension,
        int innerDimension,
        Fp8GemmVariant variant,
        ReadOnlySpan<float> bias = default)
    {
        RequireMatrix(left, rows, innerDimension, nameof(left));
        RequireMatrix(right, outputDimension, innerDimension, nameof(right));
        if (variant == Fp8GemmVariant.Sm90Bias) RequireLength(bias, outputDimension, nameof(bias));
        var output = new float[checked(rows * outputDimension)];
        fixed (float* leftPointer = left)
        fixed (float* rightPointer = right)
        fixed (float* biasPointer = bias)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.Fp8GemmHost(
                leftPointer, rightPointer, biasPointer, outputPointer,
                rows, outputDimension, innerDimension, (int)variant), nameof(Fp8Gemm));
        }
        return output;
    }

    public static unsafe NvFp4Tensor QuantizeNvFp4(
        ReadOnlySpan<float> input,
        int rows,
        int columns,
        bool highNibbleFirst = true,
        int variant = 2,
        float? perTensorScale = null)
    {
        RequireMatrix(input, rows, columns, nameof(input));
        if (columns % 16 != 0) throw new ArgumentException("NVFP4 columns must be divisible by 16.", nameof(columns));
        var scale = perTensorScale ?? AmaxScale(input, 448F * 6F);
        var packed = new byte[checked(rows * columns / 2)];
        var paddedRows = ((rows + 127) / 128) * 128;
        var paddedColumns = (((columns / 16) + 3) / 4) * 4;
        var blockScales = new byte[checked(paddedRows * paddedColumns)];
        fixed (float* inputPointer = input)
        fixed (byte* packedPointer = packed)
        fixed (byte* scalesPointer = blockScales)
        {
            Check(LtxCudaNative.NvFp4QuantizeHost(
                inputPointer, packedPointer, scalesPointer, rows, columns, scale,
                highNibbleFirst ? 1 : 0, variant), nameof(QuantizeNvFp4));
        }
        return new NvFp4Tensor(packed, blockScales, rows, columns, scale, highNibbleFirst);
    }

    public static unsafe float[] DequantizeNvFp4(NvFp4Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ValidateNvFp4(tensor);
        var output = new float[checked(tensor.Rows * tensor.Columns)];
        fixed (byte* packedPointer = tensor.Packed)
        fixed (byte* scalesPointer = tensor.BlockScales)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.NvFp4DequantizeHost(
                packedPointer, scalesPointer, outputPointer,
                tensor.Rows, tensor.Columns, tensor.PerTensorScale,
                tensor.HighNibbleFirst ? 1 : 0), nameof(DequantizeNvFp4));
        }
        return output;
    }

    public static unsafe float[] NvFp4ScaledMatrixMultiply(
        NvFp4Tensor left,
        NvFp4Tensor right,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ValidateNvFp4(left);
        ValidateNvFp4(right);
        if (left.Columns != right.Columns || left.HighNibbleFirst != right.HighNibbleFirst)
            throw new ArgumentException("NVFP4 operands must share K and nibble order.", nameof(right));
        if (!bias.IsEmpty) RequireLength(bias, right.Rows, nameof(bias));
        var output = new float[checked(left.Rows * right.Rows)];
        fixed (byte* leftPointer = left.Packed)
        fixed (byte* rightPointer = right.Packed)
        fixed (byte* leftScalesPointer = left.BlockScales)
        fixed (byte* rightScalesPointer = right.BlockScales)
        fixed (float* biasPointer = bias)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.NvFp4ScaledMmHost(
                leftPointer, rightPointer, leftScalesPointer, rightScalesPointer,
                left.PerTensorScale, right.PerTensorScale,
                biasPointer, outputPointer, left.Rows, right.Rows, left.Columns,
                left.HighNibbleFirst ? 1 : 0, bias.IsEmpty ? 0 : 1), nameof(NvFp4ScaledMatrixMultiply));
        }
        return output;
    }

    public static unsafe float MultiplyScalars(float left, float right)
    {
        float output;
        Check(LtxCudaNative.MultiplyScalarsHost(left, right, &output), nameof(MultiplyScalars));
        return output;
    }

    public static unsafe float AmaxScale(ReadOnlySpan<float> input, float divisor)
    {
        if (input.IsEmpty) throw new ArgumentException("Input cannot be empty.", nameof(input));
        float output;
        fixed (float* inputPointer = input)
        {
            Check(LtxCudaNative.AmaxScaleHost(inputPointer, (nuint)input.Length, divisor, &output), nameof(AmaxScale));
        }
        return output;
    }

    public static void ConfigureGemm(bool alphaOnDevice, bool autotune)
    {
        Check(LtxCudaNative.SetGemmAlphaOnDevice(alphaOnDevice ? 1 : 0), nameof(ConfigureGemm));
        Check(LtxCudaNative.SetGemmAutotune(autotune ? 1 : 0), nameof(ConfigureGemm));
    }

    public static int GemmAlphaMode => LtxCudaNative.GemmAlphaMode();

    public static bool ProbeGemmSupport(int rows, int outputDimension, int innerDimension) =>
        LtxCudaNative.ProbeGemmSupport(rows, outputDimension, innerDimension) == 1;

    public static unsafe byte[] PackFp6(ReadOnlySpan<byte> input, int rows, int columns)
    {
        RequireMatrix(input, rows, columns, nameof(input));
        if (columns % 4 != 0) throw new ArgumentException("FP6 columns must be divisible by 4.", nameof(columns));
        var output = new byte[checked(rows * columns * 3 / 4)];
        fixed (byte* inputPointer = input)
        fixed (byte* outputPointer = output)
        {
            Check(LtxCudaNative.Fp6PackHost(inputPointer, outputPointer, rows, columns), nameof(PackFp6));
        }
        return output;
    }

    public static unsafe byte[] UnpackFp6(ReadOnlySpan<byte> input, int rows, int columns)
    {
        RequireLength(input, checked(rows * columns * 3 / 4), nameof(input));
        var output = new byte[checked(rows * columns)];
        fixed (byte* inputPointer = input)
        fixed (byte* outputPointer = output)
        {
            Check(LtxCudaNative.Fp6UnpackHost(inputPointer, outputPointer, rows, columns), nameof(UnpackFp6));
        }
        return output;
    }

    public static unsafe float[] RmsNormRope(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> cosine,
        ReadOnlySpan<float> sine,
        int rows,
        int hiddenDimension,
        ReadOnlySpan<float> weights = default,
        float epsilon = 0F)
    {
        RequireMatrix(input, rows, hiddenDimension, nameof(input));
        RequireLength(cosine, input.Length, nameof(cosine));
        RequireLength(sine, input.Length, nameof(sine));
        if (!weights.IsEmpty) RequireLength(weights, hiddenDimension, nameof(weights));
        var output = new float[input.Length];
        fixed (float* inputPointer = input)
        fixed (float* weightsPointer = weights)
        fixed (float* cosinePointer = cosine)
        fixed (float* sinePointer = sine)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.RmsNormRopeHost(
                inputPointer, weightsPointer, cosinePointer, sinePointer, outputPointer,
                rows, hiddenDimension, weights.IsEmpty ? 0 : 1, epsilon), nameof(RmsNormRope));
        }
        return output;
    }

    public static unsafe float[] RmsNormSplitRope(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> cosine,
        ReadOnlySpan<float> sine,
        int rows,
        int heads,
        int headDimension,
        float epsilon = 1e-6F)
    {
        RequireMatrix(input, rows, checked(heads * headDimension), nameof(input));
        RequireLength(weights, checked(heads * headDimension), nameof(weights));
        RequireLength(cosine, checked(rows * heads * headDimension / 2), nameof(cosine));
        RequireLength(sine, cosine.Length, nameof(sine));
        var output = new float[input.Length];
        fixed (float* inputPointer = input)
        fixed (float* weightsPointer = weights)
        fixed (float* cosinePointer = cosine)
        fixed (float* sinePointer = sine)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.RmsNormSplitRopeHost(
                inputPointer, weightsPointer, cosinePointer, sinePointer, outputPointer,
                rows, heads, headDimension, epsilon), nameof(RmsNormSplitRope));
        }
        return output;
    }

    public static unsafe RowwiseInt8Tensor QuantizeRowwiseInt8(
        ReadOnlySpan<float> input,
        int rows,
        int columns)
    {
        RequireMatrix(input, rows, columns, nameof(input));
        var values = new sbyte[input.Length];
        var scales = new float[rows];
        fixed (float* inputPointer = input)
        fixed (sbyte* outputPointer = values)
        fixed (float* scalesPointer = scales)
        {
            Check(LtxCudaNative.RowwiseInt8QuantizeHost(
                inputPointer, outputPointer, scalesPointer, rows, columns), nameof(QuantizeRowwiseInt8));
        }
        return new RowwiseInt8Tensor(values, scales, rows, columns);
    }

    public static unsafe BlockwiseFp8Tensor QuantizeBlockwiseFp8(
        ReadOnlySpan<float> input,
        int rows,
        int columns,
        int blockSize = 128,
        bool useGelu = false)
    {
        RequireMatrix(input, rows, columns, nameof(input));
        RequireBlockSize(columns, blockSize);
        var values = new byte[input.Length];
        var scales = new float[checked(rows * columns / blockSize)];
        fixed (float* inputPointer = input)
        fixed (byte* outputPointer = values)
        fixed (float* scalesPointer = scales)
        {
            Check(LtxCudaNative.BlockwiseFp8QuantizeHost(
                inputPointer, outputPointer, scalesPointer,
                rows, columns, blockSize, useGelu ? 1 : 0), nameof(QuantizeBlockwiseFp8));
        }
        return new BlockwiseFp8Tensor(values, scales, rows, columns, blockSize);
    }

    public static unsafe BlockwiseFp8Tensor QuantizeBlockwiseFp8WithNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> normScale,
        ReadOnlySpan<float> normShift,
        int rows,
        int columns,
        int blockSize = 128,
        float epsilon = 1e-5F)
    {
        RequireMatrix(input, rows, columns, nameof(input));
        RequireLength(normScale, columns, nameof(normScale));
        RequireLength(normShift, columns, nameof(normShift));
        RequireBlockSize(columns, blockSize);
        var values = new byte[input.Length];
        var scales = new float[checked(rows * columns / blockSize)];
        fixed (float* inputPointer = input)
        fixed (float* normScalePointer = normScale)
        fixed (float* normShiftPointer = normShift)
        fixed (byte* outputPointer = values)
        fixed (float* scalesPointer = scales)
        {
            Check(LtxCudaNative.BlockwiseFp8NormQuantizeHost(
                inputPointer, normScalePointer, normShiftPointer,
                outputPointer, scalesPointer, rows, columns, blockSize, epsilon), nameof(QuantizeBlockwiseFp8WithNorm));
        }
        return new BlockwiseFp8Tensor(values, scales, rows, columns, blockSize);
    }

    public static unsafe (float[] Residual, BlockwiseFp8Tensor Quantized) QuantizedRmsFma(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> y,
        ReadOnlySpan<float> z,
        int rows,
        int columns,
        int blockSize = 128)
    {
        RequireMatrix(x, rows, columns, nameof(x));
        RequireLength(y, x.Length, nameof(y));
        RequireLength(z, x.Length, nameof(z));
        RequireBlockSize(columns, blockSize);
        var residual = new float[x.Length];
        var values = new byte[x.Length];
        var scales = new float[checked(rows * columns / blockSize)];
        fixed (float* xPointer = x)
        fixed (float* yPointer = y)
        fixed (float* zPointer = z)
        fixed (float* residualPointer = residual)
        fixed (byte* outputPointer = values)
        fixed (float* scalesPointer = scales)
        {
            Check(LtxCudaNative.QuantizedRmsFmaHost(
                xPointer, yPointer, zPointer, residualPointer, outputPointer, scalesPointer,
                rows, columns, blockSize), nameof(QuantizedRmsFma));
        }
        return (residual, new BlockwiseFp8Tensor(values, scales, rows, columns, blockSize));
    }

    public static unsafe float[] GatedAttention(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gateLogits,
        int rows,
        int heads,
        int headDimension)
    {
        RequireMatrix(input, rows, checked(heads * headDimension), nameof(input));
        RequireMatrix(gateLogits, rows, heads, nameof(gateLogits));
        var output = new float[input.Length];
        fixed (float* inputPointer = input)
        fixed (float* gatePointer = gateLogits)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.GatedAttentionHost(
                inputPointer, gatePointer, outputPointer, rows, heads, headDimension), nameof(GatedAttention));
        }
        return output;
    }

    public static unsafe float[] DequantizeBlockwiseFp8(BlockwiseFp8Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        RequireMatrix(tensor.Values, tensor.Rows, tensor.Columns, nameof(tensor));
        RequireLength(tensor.Scales, checked(tensor.Rows * tensor.Columns / tensor.BlockSize), nameof(tensor));
        var output = new float[tensor.Values.Length];
        fixed (byte* inputPointer = tensor.Values)
        fixed (float* scalesPointer = tensor.Scales)
        fixed (float* outputPointer = output)
        {
            Check(LtxCudaNative.BlockwiseFp8DequantizeHost(
                inputPointer, scalesPointer, outputPointer,
                tensor.Rows, tensor.Columns, tensor.BlockSize), nameof(DequantizeBlockwiseFp8));
        }
        return output;
    }

    public static QuantizedLoraFusion FuseNvFp4Lora(
        NvFp4Tensor weight,
        ReadOnlySpan<float> factorA,
        ReadOnlySpan<float> factorB,
        int rank,
        float strength)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ValidateNvFp4(weight);
        RequireMatrix(factorA, rank, weight.Columns, nameof(factorA));
        RequireMatrix(factorB, weight.Rows, rank, nameof(factorB));
        var merged = DequantizeNvFp4(weight);
        for (var row = 0; row < weight.Rows; row++)
        {
            for (var column = 0; column < weight.Columns; column++)
            {
                var sum = 0F;
                for (var inner = 0; inner < rank; inner++)
                {
                    var scaledB = RoundBFloat16(RoundBFloat16(factorB[row * rank + inner]) * strength);
                    var a = RoundBFloat16(factorA[inner * weight.Columns + column]);
                    sum += scaledB * a;
                }
                merged[row * weight.Columns + column] = RoundBFloat16(
                    RoundBFloat16(merged[row * weight.Columns + column]) + RoundBFloat16(sum));
            }
        }
        var fused = QuantizeNvFp4(merged, weight.Rows, weight.Columns, weight.HighNibbleFirst);
        return new QuantizedLoraFusion(weight.DeepClone(), fused);
    }

    private static float RoundBFloat16(float value)
    {
        if (float.IsNaN(value) || float.IsInfinity(value)) return value;
        var bits = BitConverter.SingleToUInt32Bits(value);
        bits += 0x7fffU + ((bits >> 16) & 1U);
        return BitConverter.UInt32BitsToSingle(bits & 0xffff0000U);
    }

    private static void Check(int status, string operation)
    {
        if (status != 0) throw new LtxCudaException(operation, status);
    }

    private static void ValidateNvFp4(NvFp4Tensor tensor)
    {
        RequireLength(tensor.Packed, checked(tensor.Rows * tensor.Columns / 2), nameof(tensor));
        RequireLength(tensor.BlockScales, checked(tensor.PaddedScaleRows * tensor.PaddedScaleColumns), nameof(tensor));
        if (tensor.Columns % 16 != 0 || tensor.Rows <= 0 || tensor.PerTensorScale < 0F)
            throw new ArgumentException("Invalid NVFP4 tensor metadata.", nameof(tensor));
    }

    private static void RequireBlockSize(int columns, int blockSize)
    {
        if (blockSize <= 0 || columns % blockSize != 0)
            throw new ArgumentException("Columns must be divisible by a positive block size.", nameof(blockSize));
    }

    private static void RequireSameLength<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, string parameterName)
    {
        if (left.Length != right.Length) throw new ArgumentException("Lengths must match.", parameterName);
    }

    private static void RequireMatrix<T>(ReadOnlySpan<T> values, int rows, int columns, string parameterName)
    {
        if (rows <= 0 || columns <= 0) throw new ArgumentOutOfRangeException(parameterName);
        RequireLength(values, checked(rows * columns), parameterName);
    }

    private static void RequireLength<T>(ReadOnlySpan<T> values, int expected, string parameterName)
    {
        if (values.Length != expected)
            throw new ArgumentException($"Expected {expected} values, received {values.Length}.", parameterName);
    }
}

public static partial class LtxCudaNative
{
    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_kernel_count")]
    internal static partial int KernelCount();

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_kernel_name")]
    internal static partial nint KernelName(int index);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_fused_add_round_f32_host")]
    internal static unsafe partial int FusedAddRoundHost(
        float* delta, float* weight, float* output, nuint elementCount, uint seed);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_na3d_f32_host")]
    internal static unsafe partial int Na3dHost(
        float* query, float* key, float* value, float* output,
        int batch, int time, int height, int width, int heads, int headDimension,
        int kernelTime, int kernelHeight, int kernelWidth,
        int causalTime, int causalHeight, int causalWidth, float scale);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_swiglu_up_mul_f32_host")]
    internal static unsafe partial int SwiGluUpMultiplyHost(
        float* input, float* upWeight, float* gate, float* output,
        int rows, int inputDimension, int outputDimension);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_swiglu_gate_up_f32_host")]
    internal static unsafe partial int SwiGluGateUpHost(
        float* input, float* gateWeight, float* upWeight, float* output,
        int rows, int inputDimension, int outputDimension);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_fp8_gemm_f32_host")]
    internal static unsafe partial int Fp8GemmHost(
        float* left, float* right, float* bias, float* output,
        int rows, int outputDimension, int innerDimension, int variant);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_nvfp4_quantize_f32_host")]
    internal static unsafe partial int NvFp4QuantizeHost(
        float* input, byte* packed, byte* blockScales,
        int rows, int columns, float perTensorScale, int highNibbleFirst, int variant);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_nvfp4_dequantize_f32_host")]
    internal static unsafe partial int NvFp4DequantizeHost(
        byte* packed, byte* blockScales, float* output,
        int rows, int columns, float perTensorScale, int highNibbleFirst);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_nvfp4_scaled_mm_f32_host")]
    internal static unsafe partial int NvFp4ScaledMmHost(
        byte* left, byte* right, byte* leftScales, byte* rightScales,
        float leftPerTensorScale, float rightPerTensorScale,
        float* bias, float* output, int rows, int outputDimension, int innerDimension,
        int highNibbleFirst, int hasBias);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_mul_scalars_f32_host")]
    internal static unsafe partial int MultiplyScalarsHost(float left, float right, float* output);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_amax_scale_f32_host")]
    internal static unsafe partial int AmaxScaleHost(float* input, nuint count, float divisor, float* output);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_gemm_alpha_mode")]
    internal static partial int GemmAlphaMode();

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_set_gemm_alpha_on_device")]
    internal static partial int SetGemmAlphaOnDevice(int enabled);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_set_gemm_autotune")]
    internal static partial int SetGemmAutotune(int enabled);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_probe_gemm_support")]
    internal static partial int ProbeGemmSupport(int rows, int outputDimension, int innerDimension);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_fp6_pack_host")]
    internal static unsafe partial int Fp6PackHost(byte* input, byte* output, int rows, int columns);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_fp6_unpack_host")]
    internal static unsafe partial int Fp6UnpackHost(byte* input, byte* output, int rows, int columns);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_rms_norm_rope_f32_host")]
    internal static unsafe partial int RmsNormRopeHost(
        float* input, float* weights, float* cosine, float* sine, float* output,
        int rows, int hiddenDimension, int hasWeights, float epsilon);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_rms_norm_split_rope_f32_host")]
    internal static unsafe partial int RmsNormSplitRopeHost(
        float* input, float* weights, float* cosine, float* sine, float* output,
        int rows, int heads, int headDimension, float epsilon);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_rowwise_int8_quantize_f32_host")]
    internal static unsafe partial int RowwiseInt8QuantizeHost(
        float* input, sbyte* output, float* scales, int rows, int columns);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_blockwise_fp8_quantize_f32_host")]
    internal static unsafe partial int BlockwiseFp8QuantizeHost(
        float* input, byte* output, float* scales,
        int rows, int columns, int blockSize, int useGelu);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_blockwise_fp8_norm_quantize_f32_host")]
    internal static unsafe partial int BlockwiseFp8NormQuantizeHost(
        float* input, float* normScale, float* normShift,
        byte* output, float* scales, int rows, int columns, int blockSize, float epsilon);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_quant_rms_fma_f32_host")]
    internal static unsafe partial int QuantizedRmsFmaHost(
        float* x, float* y, float* z, float* residual,
        byte* output, float* scales, int rows, int columns, int blockSize);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_gated_attention_f32_host")]
    internal static unsafe partial int GatedAttentionHost(
        float* input, float* gateLogits, float* output,
        int rows, int heads, int headDimension);

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_blockwise_fp8_dequantize_f32_host")]
    internal static unsafe partial int BlockwiseFp8DequantizeHost(
        byte* input, float* scales, float* output,
        int rows, int columns, int blockSize);
}
