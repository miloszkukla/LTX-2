using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public static class TorchSharpRuntime
{
    private static readonly object Sync = new();
    private static nint nativeHandle;
    private static string? nativePath;
    private static UseMathSdpOnlyDelegate? useMathSdpOnly;
    private static RmsNormDelegate? rmsNorm;

    public static nint NativeHandle => nativeHandle != 0
        ? nativeHandle
        : throw new InvalidOperationException("The LTX TorchSharp backend is not initialized.");

    public static void UseMathSdpOnly()
    {
        (useMathSdpOnly ?? throw new InvalidOperationException("The LTX TorchSharp backend is not initialized."))();
    }

    public static Tensor RmsNorm(Tensor input, Tensor? weight = null, double epsilon = 1e-6)
    {
        ArgumentNullException.ThrowIfNull(input);
        var result = (rmsNorm ?? throw new InvalidOperationException("The LTX TorchSharp backend is not initialized."))(
            input.Handle,
            weight?.Handle ?? 0,
            epsilon);
        return result != 0
            ? Tensor.UnsafeCreateTensor(result)
            : throw new ExternalException("The native LTX RMSNorm operation failed.");
    }

    public static void Initialize(string nativeLibraryPath)
    {
        var fullPath = Path.GetFullPath(nativeLibraryPath);
        lock (Sync)
        {
            if (nativeHandle != 0)
            {
                if (!string.Equals(nativePath, fullPath, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException("TorchSharp was already initialized with another ABI bridge.");
                }
                return;
            }

            var loadedHandle = NativeLibrary.Load(fullPath);
            try
            {
                NativeLibrary.SetDllImportResolver(typeof(torch).Assembly, ResolveLibrary);
                nativePath = fullPath;
                nativeHandle = loadedHandle;
                useMathSdpOnly = Load<UseMathSdpOnlyDelegate>(loadedHandle, "THSLtx_use_math_sdp_only");
                rmsNorm = Load<RmsNormDelegate>(loadedHandle, "THSLtx_rms_norm");
            }
            catch
            {
                NativeLibrary.Free(loadedHandle);
                throw;
            }
        }
    }

    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        return string.Equals(libraryName, "LibTorchSharp", StringComparison.Ordinal)
            ? nativeHandle
            : 0;
    }

    private static T Load<T>(nint handle, string symbol) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(handle, symbol));

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void UseMathSdpOnlyDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint RmsNormDelegate(nint input, nint weight, double epsilon);
}
