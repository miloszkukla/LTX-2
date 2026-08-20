using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp;

namespace Ltx.Core;

public static class TorchSharpRuntime
{
    private static readonly object Sync = new();
    private static nint nativeHandle;
    private static string? nativePath;

    public static nint NativeHandle => nativeHandle != 0
        ? nativeHandle
        : throw new InvalidOperationException("The LTX TorchSharp backend is not initialized.");

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
}
