using System.Collections.ObjectModel;
using System.Text.Json;

namespace Ltx.Core;

public sealed class CheckpointMetadata
{
    private readonly IReadOnlyDictionary<string, string> raw;

    internal CheckpointMetadata(IReadOnlyDictionary<string, string> raw)
    {
        this.raw = new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(raw, StringComparer.Ordinal));
    }

    public IReadOnlyDictionary<string, string> Raw => raw;

    public bool TryGetParsed(string key, out JsonElement value)
    {
        if (!raw.TryGetValue(key, out var text))
        {
            value = default;
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(text);
            value = document.RootElement.Clone();
        }
        catch (JsonException)
        {
            value = JsonSerializer.SerializeToElement(text);
        }

        return true;
    }

    public JsonElement? ModelConfig
    {
        get
        {
            return TryGetParsed("config", out var config) && config.ValueKind == JsonValueKind.Object
                ? config
                : null;
        }
    }

    public string? ModelVersion => raw.GetValueOrDefault("model_version");
}
