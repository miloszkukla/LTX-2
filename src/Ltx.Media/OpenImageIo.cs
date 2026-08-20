using System.Runtime.InteropServices;

namespace Ltx.Media;

public static partial class OpenImageIo
{
    public static string RuntimeVersion
    {
        get
        {
            var pointer = NativeVersion();
            return pointer == IntPtr.Zero ? "unknown" : Marshal.PtrToStringUTF8(pointer) ?? "unknown";
        }
    }

    public static FloatImage ReadExr(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("EXR input does not exist.", path);
        }
        if (NativeImageInfo(path, out var width, out var height, out var channels) != 0)
        {
            throw NativeFailure("read EXR metadata");
        }
        var pixels = new float[checked(width * height * channels)];
        if (NativeReadFloat(path, pixels, pixels.Length) != 0)
        {
            throw NativeFailure("read EXR pixels");
        }
        return new FloatImage(pixels, width, height, channels);
    }

    public static void WriteExr(string path, FloatImage image)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(image);
        if (image.Width <= 0 || image.Height <= 0 || image.Channels is < 1 or > 4 ||
            image.Pixels.Length != checked(image.Width * image.Height * image.Channels))
        {
            throw new ArgumentOutOfRangeException(nameof(image), "EXR image layout is inconsistent.");
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        if (NativeWriteFloat(path, image.Width, image.Height, image.Channels, image.Pixels) != 0)
        {
            throw NativeFailure("write EXR pixels");
        }
    }

    private static InvalidDataException NativeFailure(string operation)
    {
        var pointer = NativeLastError();
        var detail = pointer == IntPtr.Zero ? "unknown OpenImageIO failure" : Marshal.PtrToStringUTF8(pointer);
        return new InvalidDataException($"Could not {operation}: {detail}");
    }

    [LibraryImport("ltx_oiio", EntryPoint = "ltx_oiio_version")]
    private static partial IntPtr NativeVersion();

    [LibraryImport("ltx_oiio", EntryPoint = "ltx_oiio_last_error")]
    private static partial IntPtr NativeLastError();

    [LibraryImport("ltx_oiio", EntryPoint = "ltx_oiio_image_info", StringMarshalling = StringMarshalling.Utf8)]
    private static partial int NativeImageInfo(string path, out int width, out int height, out int channels);

    [LibraryImport("ltx_oiio", EntryPoint = "ltx_oiio_read_float", StringMarshalling = StringMarshalling.Utf8)]
    private static partial int NativeReadFloat(string path, [Out] float[] pixels, int elementCapacity);

    [LibraryImport("ltx_oiio", EntryPoint = "ltx_oiio_write_float", StringMarshalling = StringMarshalling.Utf8)]
    private static partial int NativeWriteFloat(
        string path,
        int width,
        int height,
        int channels,
        [In] float[] pixels);
}
