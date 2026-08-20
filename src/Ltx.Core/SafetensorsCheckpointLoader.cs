using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public sealed class SafetensorsCheckpointLoader
{
    public CheckpointMetadata ReadMetadata(string path)
    {
        return new CheckpointMetadata(SafeTensorFile.ReadMetadata(path));
    }

    public StateDictionary Load(
        string path,
        StateDictionaryOperations? operations = null,
        Device? device = null) => Load([path], operations, device);

    public StateDictionary Load(
        IEnumerable<string> paths,
        StateDictionaryOperations? operations = null,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var pathList = paths.ToArray();
        if (pathList.Length == 0)
        {
            throw new ArgumentException("At least one checkpoint shard is required.", nameof(paths));
        }

        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var sizes = new Dictionary<string, long>(StringComparer.Ordinal);
        try
        {
            foreach (var path in pathList)
            {
                var file = SafeTensorFile.Load(path);
                foreach (var (sourceName, safeTensor) in file.Tensors.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                {
                    var targetName = operations is null ? sourceName : operations.ApplyToKey(sourceName);
                    if (targetName is null)
                    {
                        continue;
                    }

                    var tensor = safeTensor.ToTorchTensor(device);
                    if (tensors.Remove(targetName, out var previous))
                    {
                        previous.Dispose();
                    }

                    tensors[targetName] = tensor;
                    sizes[targetName] = safeTensor.Data.Length;
                }
            }

            return new StateDictionary(tensors, sizes.Values.Sum());
        }
        catch
        {
            foreach (var tensor in tensors.Values)
            {
                tensor.Dispose();
            }

            throw;
        }
    }
}
