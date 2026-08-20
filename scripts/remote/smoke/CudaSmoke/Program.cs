using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 1)
{
    Console.Error.WriteLine("usage: CudaSmoke <absolute-path-to-libltx_cuda_smoke.so>");
    return 2;
}

var nativeLibraries = new[]
{
    "libavcodec.so",
    "libavformat.so",
    "libOpenImageIO.so",
    "libopencv_core.so",
};

foreach (var library in nativeLibraries)
{
    if (!NativeLibrary.TryLoad(library, out var systemHandle))
    {
        Console.Error.WriteLine($"P/Invoke load failed: {library}");
        return 3;
    }
    NativeLibrary.Free(systemHandle);
}

using (var input = tensor(new[] { 6.0f, 7.0f }))
using (var output = input.matmul(input))
{
    var value = output.item<float>();
    if (MathF.Abs(value - 85.0f) > 1e-5f)
    {
        Console.Error.WriteLine($"TorchSharp smoke mismatch: {value}");
        return 4;
    }
}

var handle = NativeLibrary.Load(args[0]);
try
{
    var export = NativeLibrary.GetExport(handle, "ltx_cuda_smoke");
    var smoke = Marshal.GetDelegateForFunctionPointer<CudaSmokeDelegate>(export);
    var status = smoke(21.0f, out var output, out var computeCapability);
    if (status != 0 || MathF.Abs(output - 42.0f) > 1e-5f)
    {
        Console.Error.WriteLine($"CUDA native smoke failed: status={status}, output={output}");
        return 5;
    }
    Console.WriteLine($"TorchSharp=pass PInvoke=pass CUDA=pass SASS=sm_{computeCapability}");
}
finally
{
    NativeLibrary.Free(handle);
}

return 0;

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate int CudaSmokeDelegate(float input, out float output, out int computeCapability);
