using System.Runtime.InteropServices;
using Ltx.AbiSmoke;
using Ltx.Core;
using Ltx.Cuda;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: Ltx.AbiSmoke <libLibTorchSharp.so> <libltx_cuda.so>");
    return 2;
}

TorchSharpRuntime.Initialize(args[0]);
using var ltxCudaLibrary = new NativeLibraryHandle(args[1]);
TorchSharpAbi.Validate(TorchSharpRuntime.NativeHandle);
LtxCudaNative.ValidateAbi();

InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available())
{
    Console.Error.WriteLine("TorchSharp CUDA backend is unavailable.");
    return 3;
}

using var input = tensor(new[] { 1.0f, -2.0f, 3.5f }, device: CUDA);
using var transformed = input.mul(2.0f).add(0.5f);
using var actualCpu = transformed.cpu();
var actual = actualCpu.data<float>().ToArray();
var expected = new[] { 2.5f, -3.5f, 7.5f };
for (var index = 0; index < expected.Length; index++)
{
    if (MathF.Abs(actual[index] - expected[index]) > 1e-5f)
    {
        Console.Error.WriteLine($"TorchSharp CUDA mismatch at {index}: {actual[index]}.");
        return 4;
    }
}

var nativeOutput = new float[expected.Length];
LtxCudaNative.Affine(new[] { 1.0f, -2.0f, 3.5f }, nativeOutput, 2.0f, 0.5f);
for (var index = 0; index < expected.Length; index++)
{
    if (MathF.Abs(nativeOutput[index] - expected[index]) > 1e-5f)
    {
        Console.Error.WriteLine($"libltx_cuda mismatch at {index}: {nativeOutput[index]}.");
        return 5;
    }
}

var capability = LtxCudaNative.DeviceComputeCapability();
Console.WriteLine($"LTX_ABI=1.0 TorchSharp={AbiContract.TorchSharpManagedVersion} PyTorch={AbiContract.PyTorchVersion} CUDA={AbiContract.CudaDisplayVersion} GPU=sm_{capability} native=pass");
return 0;

internal sealed class NativeLibraryHandle : IDisposable
{
    private readonly nint handle;

    public NativeLibraryHandle(string path)
    {
        handle = NativeLibrary.Load(Path.GetFullPath(path));
    }

    public void Dispose() => NativeLibrary.Free(handle);
}
