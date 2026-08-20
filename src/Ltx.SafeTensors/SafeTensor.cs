using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.SafeTensors;

public sealed class SafeTensor
{
    private const long MaximumDimension = 281_474_976_710_655;
    private readonly byte[] data;

    public SafeTensor(SafeTensorDType dtype, IEnumerable<long> shape, ReadOnlySpan<byte> data)
        : this(dtype, shape, data.ToArray(), takeOwnership: true)
    {
    }

    internal SafeTensor(SafeTensorDType dtype, IEnumerable<long> shape, byte[] data, bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(data);
        DType = dtype;
        Shape = shape.ToArray();
        ValidateShape(Shape);

        var requiredLength = GetRequiredByteLength(dtype, Shape);
        if (requiredLength != data.Length)
        {
            throw new ArgumentException(
                $"Tensor payload is {data.Length} bytes; dtype {dtype.ToFormatCode()} and shape " +
                $"[{string.Join(",", Shape)}] require {requiredLength} bytes.",
                nameof(data));
        }

        this.data = takeOwnership ? data : data.ToArray();
    }

    public SafeTensorDType DType { get; }

    public long[] Shape { get; }

    public ReadOnlyMemory<byte> Data => data;

    public long ElementCount => GetElementCount(Shape);

    internal ReadOnlySpan<byte> DataSpan => data;

    public Tensor ToTorchTensor(Device? device = null)
    {
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("Safetensors currently requires a little-endian host.");
        }

        return DType switch
        {
            SafeTensorDType.Bool => tensor(data.Select(value => value != 0).ToArray(), Shape, device: device),
            SafeTensorDType.UInt8 => tensor(data.ToArray(), Shape, device: device),
            SafeTensorDType.Int8 => tensor(MemoryMarshal.Cast<byte, sbyte>(data).ToArray(), Shape, device: device),
            SafeTensorDType.Int16 => tensor(MemoryMarshal.Cast<byte, short>(data).ToArray(), Shape, device: device),
            SafeTensorDType.Int32 => tensor(MemoryMarshal.Cast<byte, int>(data).ToArray(), Shape, device: device),
            SafeTensorDType.Int64 => tensor(MemoryMarshal.Cast<byte, long>(data).ToArray(), Shape, device: device),
            SafeTensorDType.Float16 => tensor(MemoryMarshal.Cast<byte, Half>(data).ToArray(), Shape, device: device),
            SafeTensorDType.BFloat16 => tensor(ToBFloat16Array(data), Shape, device: device),
            SafeTensorDType.Float32 => tensor(MemoryMarshal.Cast<byte, float>(data).ToArray(), Shape, device: device),
            SafeTensorDType.Float64 => tensor(MemoryMarshal.Cast<byte, double>(data).ToArray(), Shape, device: device),
            _ => throw new NotSupportedException(
                $"TorchSharp 0.107.0 cannot materialize safetensors dtype {DType.ToFormatCode()}; raw read/write remains supported."),
        };
    }

    public static SafeTensor FromTorchTensor(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        using var detached = tensor.detach();
        using var cpu = detached.cpu();
        using var contiguous = cpu.contiguous();

        var shape = contiguous.shape.ToArray();
        if (contiguous.numel() == 0)
        {
            return new SafeTensor(FromTorchDType(contiguous.dtype), shape, []);
        }

        return contiguous.dtype switch
        {
            ScalarType.Bool => new SafeTensor(
                SafeTensorDType.Bool,
                shape,
                contiguous.data<bool>().ToArray().Select(value => value ? (byte)1 : (byte)0).ToArray()),
            ScalarType.Byte => new SafeTensor(SafeTensorDType.UInt8, shape, contiguous.data<byte>().ToArray()),
            ScalarType.Int8 => FromUnmanaged(SafeTensorDType.Int8, shape, contiguous.data<sbyte>().ToArray()),
            ScalarType.Int16 => FromUnmanaged(SafeTensorDType.Int16, shape, contiguous.data<short>().ToArray()),
            ScalarType.Int32 => FromUnmanaged(SafeTensorDType.Int32, shape, contiguous.data<int>().ToArray()),
            ScalarType.Int64 => FromUnmanaged(SafeTensorDType.Int64, shape, contiguous.data<long>().ToArray()),
            ScalarType.Float16 => FromUnmanaged(SafeTensorDType.Float16, shape, contiguous.data<Half>().ToArray()),
            ScalarType.BFloat16 => FromUnmanaged(
                SafeTensorDType.BFloat16,
                shape,
                contiguous.data<BFloat16>().ToArray()),
            ScalarType.Float32 => FromUnmanaged(SafeTensorDType.Float32, shape, contiguous.data<float>().ToArray()),
            ScalarType.Float64 => FromUnmanaged(SafeTensorDType.Float64, shape, contiguous.data<double>().ToArray()),
            _ => throw new NotSupportedException($"TorchSharp dtype {contiguous.dtype} cannot be written as safetensors."),
        };
    }

    internal static long GetRequiredByteLength(SafeTensorDType dtype, IReadOnlyList<long> shape)
    {
        var elements = GetElementCount(shape);
        var bits = checked(elements * dtype.BitsPerElement());
        return checked((bits + 7) / 8);
    }

    private static long GetElementCount(IReadOnlyList<long> shape)
    {
        long result = 1;
        foreach (var dimension in shape)
        {
            result = checked(result * dimension);
        }

        return result;
    }

    private static void ValidateShape(IEnumerable<long> shape)
    {
        foreach (var dimension in shape)
        {
            if (dimension < 0 || dimension > MaximumDimension)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(shape),
                    dimension,
                    $"Safetensors dimensions must be between 0 and {MaximumDimension}.");
            }
        }
    }

    private static BFloat16[] ToBFloat16Array(ReadOnlySpan<byte> bytes)
    {
        var source = MemoryMarshal.Cast<byte, ushort>(bytes);
        var result = new BFloat16[source.Length];
        for (var index = 0; index < source.Length; index++)
        {
            result[index] = BFloat16.FromRawValue(source[index]);
        }

        return result;
    }

    private static SafeTensor FromUnmanaged<T>(SafeTensorDType dtype, long[] shape, T[] values)
        where T : unmanaged => new(dtype, shape, MemoryMarshal.AsBytes(values.AsSpan()));

    private static SafeTensorDType FromTorchDType(ScalarType dtype) => dtype switch
    {
        ScalarType.Bool => SafeTensorDType.Bool,
        ScalarType.Byte => SafeTensorDType.UInt8,
        ScalarType.Int8 => SafeTensorDType.Int8,
        ScalarType.Int16 => SafeTensorDType.Int16,
        ScalarType.Int32 => SafeTensorDType.Int32,
        ScalarType.Int64 => SafeTensorDType.Int64,
        ScalarType.Float16 => SafeTensorDType.Float16,
        ScalarType.BFloat16 => SafeTensorDType.BFloat16,
        ScalarType.Float32 => SafeTensorDType.Float32,
        ScalarType.Float64 => SafeTensorDType.Float64,
        _ => throw new NotSupportedException($"TorchSharp dtype {dtype} cannot be written as safetensors."),
    };
}
