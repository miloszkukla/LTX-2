using System.Buffers;
using System.Buffers.Binary;
using System.Text.Json;

namespace Ltx.SafeTensors;

public sealed class SafeTensorFile
{
    private const int MaximumHeaderBytes = 100_000_000;
    private const long MaximumOffset = 281_474_976_710_655;

    public SafeTensorFile(
        IReadOnlyDictionary<string, SafeTensor> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        Tensors = new Dictionary<string, SafeTensor>(tensors, StringComparer.Ordinal);
        Metadata = metadata is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(metadata, StringComparer.Ordinal);

        if (Tensors.ContainsKey("__metadata__"))
        {
            throw new ArgumentException("'__metadata__' is reserved for safetensors metadata.", nameof(tensors));
        }
    }

    public IReadOnlyDictionary<string, SafeTensor> Tensors { get; }

    public IReadOnlyDictionary<string, string> Metadata { get; }

    public static SafeTensorFile Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
        var parsed = ReadHeader(stream);
        var payloadStart = checked((long)sizeof(ulong) + parsed.HeaderLength);
        var payloadLength = stream.Length - payloadStart;
        ValidateLayout(parsed.Descriptors, payloadLength);

        var tensors = new Dictionary<string, SafeTensor>(parsed.Descriptors.Count, StringComparer.Ordinal);
        foreach (var descriptor in parsed.Descriptors)
        {
            var length = checked(descriptor.End - descriptor.Begin);
            if (length > Array.MaxLength)
            {
                throw new InvalidDataException(
                    $"Tensor '{descriptor.Name}' is {length} bytes and cannot be represented by a managed byte array.");
            }

            stream.Position = checked(payloadStart + descriptor.Begin);
            var bytes = new byte[checked((int)length)];
            stream.ReadExactly(bytes);
            tensors.Add(descriptor.Name, new SafeTensor(descriptor.DType, descriptor.Shape, bytes));
        }

        return new SafeTensorFile(tensors, parsed.Metadata);
    }

    public static IReadOnlyDictionary<string, string> ReadMetadata(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.SequentialScan);
        var parsed = ReadHeader(stream);
        var payloadLength = stream.Length - sizeof(ulong) - parsed.HeaderLength;
        ValidateLayout(parsed.Descriptors, payloadLength);
        return new Dictionary<string, string>(parsed.Metadata, StringComparer.Ordinal);
    }

    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var ordered = Tensors.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray();
        var offsets = new Dictionary<string, (long Begin, long End)>(StringComparer.Ordinal);
        long cursor = 0;
        foreach (var (name, tensor) in ordered)
        {
            if (name == "__metadata__")
            {
                throw new InvalidDataException("'__metadata__' is reserved for safetensors metadata.");
            }

            var end = checked(cursor + tensor.Data.Length);
            if (end > MaximumOffset)
            {
                throw new InvalidDataException($"Safetensors payload exceeds the 48-bit format limit at '{name}'.");
            }

            offsets.Add(name, (cursor, end));
            cursor = end;
        }

        var headerBuffer = new ArrayBufferWriter<byte>();
        using (var json = new Utf8JsonWriter(headerBuffer, new JsonWriterOptions { Indented = false }))
        {
            json.WriteStartObject();
            if (Metadata.Count > 0)
            {
                json.WritePropertyName("__metadata__");
                json.WriteStartObject();
                foreach (var (key, value) in Metadata.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                {
                    json.WriteString(key, value);
                }

                json.WriteEndObject();
            }

            foreach (var (name, tensor) in ordered)
            {
                var (begin, end) = offsets[name];
                json.WritePropertyName(name);
                json.WriteStartObject();
                json.WriteString("dtype", tensor.DType.ToFormatCode());
                json.WritePropertyName("shape");
                json.WriteStartArray();
                foreach (var dimension in tensor.Shape)
                {
                    json.WriteNumberValue(dimension);
                }

                json.WriteEndArray();
                json.WritePropertyName("data_offsets");
                json.WriteStartArray();
                json.WriteNumberValue(begin);
                json.WriteNumberValue(end);
                json.WriteEndArray();
                json.WriteEndObject();
            }

            json.WriteEndObject();
        }

        var padding = 8 - (headerBuffer.WrittenCount % 8);
        var headerLength = checked(headerBuffer.WrittenCount + padding);
        if (headerLength > MaximumHeaderBytes)
        {
            throw new InvalidDataException($"Safetensors header exceeds {MaximumHeaderBytes} bytes.");
        }

        var fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        var temporaryPath = fullPath + $".tmp.{Guid.NewGuid():N}";
        try
        {
            using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                Span<byte> prefix = stackalloc byte[sizeof(ulong)];
                BinaryPrimitives.WriteUInt64LittleEndian(prefix, checked((ulong)headerLength));
                stream.Write(prefix);
                stream.Write(headerBuffer.WrittenSpan);
                Span<byte> spaces = stackalloc byte[8];
                spaces.Fill((byte)' ');
                stream.Write(spaces[..padding]);
                foreach (var (_, tensor) in ordered)
                {
                    stream.Write(tensor.DataSpan);
                }

                stream.Flush(true);
            }

            File.Move(temporaryPath, fullPath, true);
        }
        finally
        {
            File.Delete(temporaryPath);
        }
    }

    private static void ReadMetadata(JsonElement element, IDictionary<string, string> metadata)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("Safetensors __metadata__ must be an object.");
        }

        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.String || !metadata.TryAdd(property.Name, property.Value.GetString()!))
            {
                throw new InvalidDataException($"Safetensors metadata '{property.Name}' must be a unique string value.");
            }
        }
    }

    private static ParsedHeader ReadHeader(FileStream stream)
    {
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

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        var descriptors = new List<TensorDescriptor>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in document.RootElement.EnumerateObject())
        {
            if (!names.Add(property.Name))
            {
                throw new InvalidDataException($"Duplicate safetensors header key '{property.Name}'.");
            }

            if (property.NameEquals("__metadata__"))
            {
                ReadMetadata(property.Value, metadata);
                continue;
            }

            descriptors.Add(ReadDescriptor(property.Name, property.Value));
        }

        return new ParsedHeader(headerLength, metadata, descriptors);
    }

    private static TensorDescriptor ReadDescriptor(string name, JsonElement element)
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
        var expectedLength = SafeTensor.GetRequiredByteLength(dtype, shape);
        if (end - begin != expectedLength)
        {
            throw new InvalidDataException(
                $"Tensor '{name}' offsets describe {end - begin} bytes but dtype/shape require {expectedLength}.");
        }

        return new TensorDescriptor(name, dtype, shape, begin, end);
    }

    private static long ReadNatural(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt64(out var value) ||
            value < 0 || value > MaximumOffset)
        {
            throw new InvalidDataException($"Safetensors sizes and offsets must be natural integers no greater than {MaximumOffset}.");
        }

        return value;
    }

    private static void ValidateLayout(IEnumerable<TensorDescriptor> descriptors, long payloadLength)
    {
        long cursor = 0;
        foreach (var descriptor in descriptors.OrderBy(item => item.Begin).ThenBy(item => item.End))
        {
            if (descriptor.Begin != cursor)
            {
                throw new InvalidDataException(
                    $"Safetensors payload must be fully indexed without holes or overlap; '{descriptor.Name}' begins at {descriptor.Begin}, expected {cursor}.");
            }

            cursor = descriptor.End;
        }

        if (cursor != payloadLength)
        {
            throw new InvalidDataException(
                $"Safetensors tensor offsets index {cursor} payload bytes but the file contains {payloadLength}.");
        }
    }

    private sealed record TensorDescriptor(
        string Name,
        SafeTensorDType DType,
        long[] Shape,
        long Begin,
        long End);

    private sealed record ParsedHeader(
        int HeaderLength,
        Dictionary<string, string> Metadata,
        List<TensorDescriptor> Descriptors);
}
