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

using var addLeft = tensor(new[] { 1.0f, -2.0f }, dtype: ScalarType.BFloat16, device: CUDA)
    .reshape(1, 1, 2);
using var addRight = tensor(new[] { 0.5f, 3.0f, -4.0f, 2.0f }, dtype: ScalarType.BFloat16, device: CUDA)
    .reshape(1, 2, 2);
using var addResult = TorchSharpRuntime.Add(addLeft, addRight);
using var addResultCpu = addResult.to(ScalarType.Float32).cpu();
var actualAdd = addResultCpu.data<float>().ToArray();
var expectedAdd = new[] { 1.5f, 1.0f, -3.0f, 0.0f };
for (var index = 0; index < expectedAdd.Length; index++)
{
    if (MathF.Abs(actualAdd[index] - expectedAdd[index]) > 1e-5f)
    {
        Console.Error.WriteLine($"Native tensor-add mismatch at {index}: {actualAdd[index]}.");
        return 6;
    }
}

using var linearInput = tensor(new[] { 1.0f, -2.0f }, dtype: ScalarType.BFloat16, device: CUDA)
    .reshape(1, 2);
using var linearWeight = tensor(new[] { 2.0f, 0.5f, -1.0f, 3.0f }, dtype: ScalarType.BFloat16, device: CUDA)
    .reshape(2, 2);
using var linearBias = tensor(new[] { 0.25f, -0.5f }, dtype: ScalarType.BFloat16, device: CUDA);
using var linearResult = TorchSharpRuntime.Linear(linearInput, linearWeight, linearBias);
using var linearResultCpu = linearResult.to(ScalarType.Float32).cpu();
var actualLinear = linearResultCpu.data<float>().ToArray();
var expectedLinear = new[] { 1.25f, -7.5f };
for (var index = 0; index < expectedLinear.Length; index++)
{
    if (MathF.Abs(actualLinear[index] - expectedLinear[index]) > 1e-5f)
    {
        Console.Error.WriteLine($"Native linear mismatch at {index}: {actualLinear[index]}.");
        return 7;
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
