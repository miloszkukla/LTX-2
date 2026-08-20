using System.Runtime.InteropServices;
using Ltx.Core;

namespace Ltx.Cuda;

public static partial class LtxCudaNative
{
    private const string LibraryName = "ltx_cuda";

    public static void ValidateAbi()
    {
        var packed = AbiVersion();
        if (packed != AbiContract.PackedVersion)
        {
            throw new InvalidOperationException($"libltx_cuda ABI mismatch: 0x{packed:x8}.");
        }

        var cuda = CompiledCudaVersion();
        if (cuda != AbiContract.CudaVersion)
        {
            throw new InvalidOperationException($"libltx_cuda CUDA mismatch: {cuda}.");
        }
    }

    public static unsafe void Affine(ReadOnlySpan<float> input, Span<float> output, float scale, float bias)
    {
        if (input.Length != output.Length)
        {
            throw new ArgumentException("Input and output lengths must match.");
        }

        if (input.IsEmpty)
        {
            return;
        }

        fixed (float* inputPointer = input)
        fixed (float* outputPointer = output)
        {
            var status = AffineHost(inputPointer, outputPointer, (nuint)input.Length, scale, bias);
            if (status != 0)
            {
                throw new LtxCudaException(nameof(Affine), status);
            }
        }
    }

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_abi_version")]
    public static partial uint AbiVersion();

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_compiled_cuda_version")]
    public static partial int CompiledCudaVersion();

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_device_compute_capability")]
    public static partial int DeviceComputeCapability();

    [LibraryImport(LibraryName, EntryPoint = "ltx_cuda_affine_f32_host")]
    private static unsafe partial int AffineHost(
        float* input,
        float* output,
        nuint elementCount,
        float scale,
        float bias);
}
