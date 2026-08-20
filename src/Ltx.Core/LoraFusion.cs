using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public sealed record LoraAdapter(StateDictionary Weights, double Strength);

public static class LoraFusion
{
    private const string LoraASuffix = ".lora_A.weight";

    public static StateDictionary Fuse(StateDictionary model, IEnumerable<LoraAdapter> adapters)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(adapters);
        var adapterList = adapters.ToArray();
        var result = model.Tensors.ToDictionary(
            pair => pair.Key,
            pair => pair.Value.clone(),
            StringComparer.Ordinal);

        try
        {
            var affectedKeys = adapterList
                .SelectMany(adapter => adapter.Weights.Tensors.Keys)
                .Where(key => key.EndsWith(LoraASuffix, StringComparison.Ordinal))
                .Select(key => key[..^LoraASuffix.Length] + ".weight")
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal);

            foreach (var weightKey in affectedKeys)
            {
                if (!model.TryGetTensor(weightKey, out var original))
                {
                    continue;
                }

                var prefix = weightKey[..^".weight".Length];
                var keyA = prefix + LoraASuffix;
                var keyB = prefix + ".lora_B.weight";
                Tensor? aggregate = null;
                try
                {
                    foreach (var adapter in adapterList)
                    {
                        if (!adapter.Weights.TryGetTensor(keyA, out var aSource) ||
                            !adapter.Weights.TryGetTensor(keyB, out var bSource))
                        {
                            continue;
                        }

                        using var a = aSource.to_type(ScalarType.BFloat16);
                        using var b = bSource.to_type(ScalarType.BFloat16);
                        using var scaledB = b.mul(adapter.Strength);
                        using var product = scaledB.matmul(a);
                        if (aggregate is null)
                        {
                            aggregate = product.clone();
                        }
                        else
                        {
                            using var previous = aggregate;
                            aggregate = previous.add(product);
                        }
                    }

                    if (aggregate is null)
                    {
                        continue;
                    }

                    using var weight = original.to_type(ScalarType.BFloat16);
                    using var sum = aggregate.add(weight);
                    var fused = sum.to_type(original.dtype);
                    result[weightKey].Dispose();
                    result[weightKey] = fused;
                }
                finally
                {
                    aggregate?.Dispose();
                }
            }

            return new StateDictionary(result, model.SizeBytes);
        }
        catch
        {
            foreach (var tensor in result.Values)
            {
                tensor.Dispose();
            }

            throw;
        }
    }

    public static StateDictionary Unfuse(StateDictionary fusedModel, IEnumerable<LoraAdapter> adapters) =>
        Fuse(fusedModel, adapters.Select(adapter => adapter with { Strength = -adapter.Strength }));
}
