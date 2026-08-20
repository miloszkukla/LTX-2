using System.Runtime.InteropServices;
using System.Text.Json;
using Ltx.Core;
using Ltx.Cuda;
using Ltx.ParityRunner;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: Ltx.ParityRunner <fixture.json> <libLibTorchSharp.so> <libltx_cuda.so>");
    return 2;
}

var fixture = JsonSerializer.Deserialize<FixtureDocument>(File.ReadAllText(args[0]))
    ?? throw new InvalidDataException("Fixture is empty.");
if (fixture.SchemaVersion != 1 || fixture.Oracle.FrameworkVersion != "2.13.0+cu132" ||
    fixture.Oracle.CudaVersion != AbiContract.CudaDisplayVersion)
{
    throw new InvalidDataException("Fixture ABI metadata does not match M1.");
}

TorchSharpRuntime.Initialize(args[1]);
using var ltxCudaLibrary = new NativeLibraryHandle(args[2]);
InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available())
{
    Console.Error.WriteLine("TorchSharp CUDA backend is unavailable.");
    return 3;
}

var passed = 0;
using (var input = tensor(fixture.TorchSharpCase.Input, fixture.TorchSharpCase.InputShape, device: CUDA))
using (var weight = tensor(fixture.TorchSharpCase.Weight, fixture.TorchSharpCase.WeightShape, device: CUDA))
using (var bias = tensor(fixture.TorchSharpCase.Bias, device: CUDA))
using (var linear = input.matmul(weight).add(bias))
using (var output = nn.functional.silu(linear))
using (var outputCpu = output.cpu())
{
    AssertClose(
        fixture.TorchSharpCase.Name,
        outputCpu.data<float>().ToArray(),
        fixture.TorchSharpCase.Expected,
        fixture.Tolerance);
    passed++;
}

var nativeOutput = new float[fixture.NativeCase.Expected.Length];
LtxCudaNative.Affine(
    fixture.NativeCase.Input,
    nativeOutput,
    fixture.NativeCase.Scale,
    fixture.NativeCase.Bias);
AssertClose(
    fixture.NativeCase.Name,
    nativeOutput,
    fixture.NativeCase.Expected,
    fixture.Tolerance);
passed++;

Console.WriteLine(
    $"Parity fixtures: {passed} passed, 0 failed; FP32 rtol={fixture.Tolerance.Relative:g}, atol={fixture.Tolerance.Absolute:g}; revision={fixture.FixtureRevision}");
return 0;

static void AssertClose(string name, float[] actual, float[] expected, Tolerance tolerance)
{
    if (actual.Length != expected.Length)
    {
        throw new InvalidDataException($"{name}: length mismatch.");
    }

    for (var index = 0; index < actual.Length; index++)
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

internal sealed class NativeLibraryHandle : IDisposable
{
    private readonly nint handle;

    public NativeLibraryHandle(string path)
    {
        handle = NativeLibrary.Load(Path.GetFullPath(path));
    }

    public void Dispose() => NativeLibrary.Free(handle);
}
