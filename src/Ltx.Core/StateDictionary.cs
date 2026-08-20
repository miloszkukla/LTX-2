using System.Collections.ObjectModel;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public sealed class StateDictionary : IDisposable
{
    private readonly Dictionary<string, Tensor> tensors;
    private readonly ReadOnlyDictionary<string, Tensor> readOnlyTensors;
    private bool disposed;

    internal StateDictionary(Dictionary<string, Tensor> tensors, long sizeBytes)
    {
        this.tensors = tensors;
        readOnlyTensors = new ReadOnlyDictionary<string, Tensor>(tensors);
        SizeBytes = sizeBytes;
        DTypes = tensors.Values.Select(tensor => tensor.dtype).ToHashSet();
    }

    public IReadOnlyDictionary<string, Tensor> Tensors => readOnlyTensors;

    public long SizeBytes { get; }

    public IReadOnlySet<ScalarType> DTypes { get; }

    public bool TryGetTensor(string key, out Tensor tensor) => tensors.TryGetValue(key, out tensor!);

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        disposed = true;
        foreach (var tensor in tensors.Values)
        {
            tensor.Dispose();
        }

        tensors.Clear();
    }
}
