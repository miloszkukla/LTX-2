using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

/// <summary>Indexed, on-demand tensor access for one or more production checkpoint shards.</summary>
public sealed class CheckpointTensorStore : IDisposable
{
    private readonly Dictionary<string, (SafeTensorIndex Index, string SourceName)> locations =
        new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> cache = new(StringComparer.Ordinal);
    private bool disposed;

    public CheckpointTensorStore(IEnumerable<string> paths, StateDictionaryOperations? operations = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var pathList = paths.Select(Path.GetFullPath).ToArray();
        if (pathList.Length == 0)
        {
            throw new ArgumentException("At least one checkpoint shard is required.", nameof(paths));
        }

        var indexes = pathList.Select(SafeTensorIndex.Open).ToArray();
        Indexes = indexes;
        foreach (var index in indexes)
        {
            foreach (var sourceName in index.Tensors.Keys.Order(StringComparer.Ordinal))
            {
                var targetName = operations is null ? sourceName : operations.ApplyToKey(sourceName);
                if (targetName is not null)
                {
                    locations[targetName] = (index, sourceName);
                }
            }
        }
        Metadata = indexes[0].Metadata;
    }

    public IReadOnlyList<SafeTensorIndex> Indexes { get; }

    public IReadOnlyDictionary<string, string> Metadata { get; }

    public IEnumerable<string> TensorNames => locations.Keys;

    public long SizeBytes => locations.Values.Sum(item => item.Index.Tensors[item.SourceName].ByteLength);

    public bool Contains(string name) => locations.ContainsKey(name);

    public SafeTensorDescriptor Describe(string name)
    {
        ThrowIfDisposed();
        var location = Resolve(name);
        return location.Index.Tensors[location.SourceName];
    }

    public Tensor Load(string name, Device device, ScalarType? dtype = null, bool cacheTensor = true)
    {
        ArgumentNullException.ThrowIfNull(device);
        ThrowIfDisposed();
        var cacheKey = dtype is null ? name : $"{name}\0{dtype.Value}";
        if (cacheTensor && cache.TryGetValue(cacheKey, out var existing))
        {
            return existing;
        }

        var location = Resolve(name);
        var loaded = location.Index.LoadTorchTensor(location.SourceName, device);
        if (dtype is { } targetDType && loaded.dtype != targetDType)
        {
            loaded = loaded.to(targetDType, disposeAfter: true);
        }
        if (cacheTensor)
        {
            loaded.DetachFromDisposeScope();
            cache.Add(cacheKey, loaded);
        }
        return loaded;
    }

    public void ClearCache()
    {
        foreach (var tensor in cache.Values)
        {
            tensor.Dispose();
        }
        cache.Clear();
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        ClearCache();
    }

    private (SafeTensorIndex Index, string SourceName) Resolve(string name) =>
        locations.TryGetValue(name, out var location)
            ? location
            : throw new KeyNotFoundException($"Checkpoint tensor '{name}' does not exist.");

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}
