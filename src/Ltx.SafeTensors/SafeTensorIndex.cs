using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.SafeTensors;

public sealed record SafeTensorDescriptor(
    string Name,
    SafeTensorDType DType,
    IReadOnlyList<long> Shape,
    long DataOffset,
    long ByteLength);

/// <summary>
/// Parses and validates a safetensors file without materializing its payload. Individual tensors
/// can then be read on demand, which is required for production checkpoints whose aggregate size
/// is larger than a single managed array and for block-streamed GPU execution.
/// </summary>
public sealed class SafeTensorIndex
{
    private const int MaximumHeaderBytes = 100_000_000;
    private const long MaximumOffset = 281_474_976_710_655;
    private readonly string path;
    private readonly IReadOnlyDictionary<string, SafeTensorDescriptor> tensors;
    private readonly IReadOnlyDictionary<string, string> metadata;

    private SafeTensorIndex(
        string path,
        Dictionary<string, SafeTensorDescriptor> tensors,
        Dictionary<string, string> metadata,
        long payloadOffset,
        long fileLength)
    {
        this.path = path;
        this.tensors = new ReadOnlyDictionary<string, SafeTensorDescriptor>(tensors);
        this.metadata = new ReadOnlyDictionary<string, string>(metadata);
        PayloadOffset = payloadOffset;
        FileLength = fileLength;
    }

    public IReadOnlyDictionary<string, SafeTensorDescriptor> Tensors => tensors;

    public IReadOnlyDictionary<string, string> Metadata => metadata;

    public long PayloadOffset { get; }

    public long FileLength { get; }

    public static SafeTensorIndex Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var fullPath = Path.GetFullPath(path);
        using var stream = new FileStream(
            fullPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            1024 * 1024,
            FileOptions.RandomAccess);
        if (stream.Length < sizeof(ulong) + 2)
        {
            throw new InvalidDataException("Safetensors file is too short to contain a header and payload.");
        }

        Span<byte> prefix = stackalloc byte[sizeof(ulong)];
        stream.ReadExactly(prefix);
        var headerLengthUnsigned = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerLengthUnsigned == 0 || headerLengthUnsigned > MaximumHeaderBytes ||
            headerLengthUnsigned > (ulong)(stream.Length - sizeof(ulong)))
        {
            throw new InvalidDataException($"Invalid safetensors header length {headerLengthUnsigned}.");
        }

        var headerLength = checked((int)headerLengthUnsigned);
        var header = new byte[headerLength];
        stream.ReadExactly(header);
        if (header[0] != (byte)'{')
        {
            throw new InvalidDataException("Safetensors header must begin with a JSON object.");
        }

        using var document = JsonDocument.Parse(header, new JsonDocumentOptions
        {
            AllowTrailingCommas = false,
            CommentHandling = JsonCommentHandling.Disallow,
            MaxDepth = 64,
        });
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("Safetensors header root must be an object.");
        }

        var payloadOffset = checked((long)sizeof(ulong) + headerLength);
        var payloadLength = checked(stream.Length - payloadOffset);
        var descriptors = new Dictionary<string, SafeTensorDescriptor>(StringComparer.Ordinal);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var property in document.RootElement.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
            {
                ReadMetadata(property.Value, metadata);
                continue;
            }

            var descriptor = ReadDescriptor(property.Name, property.Value, payloadOffset);
            if (!descriptors.TryAdd(property.Name, descriptor))
            {
                throw new InvalidDataException($"Duplicate safetensors header key '{property.Name}'.");
            }
        }

        long cursor = 0;
        foreach (var descriptor in descriptors.Values.OrderBy(item => item.DataOffset))
        {
            var relativeOffset = descriptor.DataOffset - payloadOffset;
            if (relativeOffset != cursor)
            {
                throw new InvalidDataException(
                    $"Safetensors payload must be fully indexed without holes or overlap; " +
                    $"'{descriptor.Name}' begins at {relativeOffset}, expected {cursor}.");
            }
            cursor = checked(cursor + descriptor.ByteLength);
        }
        if (cursor != payloadLength)
        {
            throw new InvalidDataException(
                $"Safetensors tensor offsets index {cursor} payload bytes but the file contains {payloadLength}.");
        }

        return new SafeTensorIndex(fullPath, descriptors, metadata, payloadOffset, stream.Length);
    }

    public SafeTensor ReadTensor(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (!tensors.TryGetValue(name, out var descriptor))
        {
            throw new KeyNotFoundException($"Safetensors tensor '{name}' does not exist.");
        }
        if (descriptor.ByteLength > Array.MaxLength)
        {
            throw new InvalidDataException(
                $"Tensor '{name}' is {descriptor.ByteLength} bytes and cannot be represented by a managed byte array.");
        }

        var bytes = GC.AllocateUninitializedArray<byte>(checked((int)descriptor.ByteLength));
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        var read = 0;
        while (read < bytes.Length)
        {
            var count = RandomAccess.Read(handle, bytes.AsSpan(read), checked(descriptor.DataOffset + read));
            if (count == 0)
            {
                throw new EndOfStreamException($"Safetensors tensor '{name}' ended after {read} of {bytes.Length} bytes.");
            }
            read += count;
        }
        return new SafeTensor(descriptor.DType, descriptor.Shape, bytes, takeOwnership: true);
    }

    public Tensor LoadTorchTensor(string name, Device? device = null) => ReadTensor(name).ToTorchTensor(device);

    private static void ReadMetadata(JsonElement element, IDictionary<string, string> target)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("Safetensors __metadata__ must be an object.");
        }
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.String || !target.TryAdd(property.Name, property.Value.GetString()!))
            {
                throw new InvalidDataException($"Safetensors metadata '{property.Name}' must be a unique string value.");
            }
        }
    }

    private static SafeTensorDescriptor ReadDescriptor(string name, JsonElement element, long payloadOffset)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Safetensors tensor '{name}' descriptor must be an object.");
        }

        string? dtypeCode = null;
        long[]? shape = null;
        long begin = -1;
        long end = -1;
        var fields = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (!fields.Add(property.Name))
            {
                throw new InvalidDataException($"Duplicate field '{property.Name}' in tensor '{name}'.");
            }
            switch (property.Name)
            {
                case "dtype" when property.Value.ValueKind == JsonValueKind.String:
                    dtypeCode = property.Value.GetString();
                    break;
                case "shape" when property.Value.ValueKind == JsonValueKind.Array:
                    shape = property.Value.EnumerateArray().Select(ReadNatural).ToArray();
                    break;
                case "data_offsets" when property.Value.ValueKind == JsonValueKind.Array:
                    var offsets = property.Value.EnumerateArray().Select(ReadNatural).ToArray();
                    if (offsets.Length != 2)
                    {
                        throw new InvalidDataException($"Tensor '{name}' data_offsets must contain exactly two values.");
                    }
                    begin = offsets[0];
                    end = offsets[1];
                    break;
                default:
                    throw new InvalidDataException($"Invalid or unsupported field '{property.Name}' in tensor '{name}'.");
            }
        }

        if (dtypeCode is null || shape is null || begin < 0 || end < begin)
        {
            throw new InvalidDataException($"Tensor '{name}' has an incomplete or invalid descriptor.");
        }
        var dtype = SafeTensorDTypeExtensions.ParseFormatCode(dtypeCode);
        var length = checked(end - begin);
        if (length != SafeTensor.GetRequiredByteLength(dtype, shape))
        {
            throw new InvalidDataException($"Tensor '{name}' offsets do not match its dtype and shape.");
        }
        return new SafeTensorDescriptor(name, dtype, shape, checked(payloadOffset + begin), length);
    }

    private static long ReadNatural(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt64(out var value) ||
            value < 0 || value > MaximumOffset)
        {
            throw new InvalidDataException(
                $"Safetensors sizes and offsets must be natural integers no greater than {MaximumOffset}.");
        }
        return value;
    }
}
