using System.Runtime.InteropServices;
using Ltx.Core;

namespace Ltx.AbiSmoke;

internal static class TorchSharpAbi
{
    public static void Validate(nint nativeHandle)
    {
        var abiVersion = Load<UInt32Delegate>(nativeHandle, "THSLtx_abi_version");
        var torchVersionExport = Load<PointerDelegate>(nativeHandle, "THSLtx_torch_version");
        var cudaVersion = Load<Int32Delegate>(nativeHandle, "THSLtx_cuda_version");
        var cxx11Abi = Load<Int32Delegate>(nativeHandle, "THSLtx_cxx11_abi");

        if (abiVersion() != AbiContract.PackedVersion)
        {
            throw new InvalidOperationException("LTX TorchSharp ABI version mismatch.");
        }

        var torchVersion = Marshal.PtrToStringUTF8(torchVersionExport()) ?? string.Empty;
        if (torchVersion != AbiContract.PyTorchVersion)
        {
            throw new InvalidOperationException($"PyTorch ABI mismatch: {torchVersion}.");
        }

        if (cudaVersion() != AbiContract.CudaVersion)
        {
            throw new InvalidOperationException($"TorchSharp CUDA ABI mismatch: {cudaVersion()}.");
        }

        if (cxx11Abi() != AbiContract.Cxx11Abi)
        {
            throw new InvalidOperationException($"TorchSharp C++ ABI mismatch: {cxx11Abi()}.");
        }
    }

    private static T Load<T>(nint nativeHandle, string name)
        where T : Delegate => Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(nativeHandle, name));

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate uint UInt32Delegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Int32Delegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint PointerDelegate();
}
