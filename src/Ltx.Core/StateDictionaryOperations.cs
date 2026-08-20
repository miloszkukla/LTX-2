namespace Ltx.Core;

public sealed record StateDictionaryMatch(string Prefix = "", string Suffix = "", string Contains = "")
{
    internal bool Matches(string key) =>
        key.StartsWith(Prefix, StringComparison.Ordinal) &&
        key.EndsWith(Suffix, StringComparison.Ordinal) &&
        (Contains.Length == 0 || key.Contains(Contains, StringComparison.Ordinal));
}

public sealed record StateDictionaryReplacement(string Content, string Replacement);

public sealed class StateDictionaryOperations
{
    private readonly StateDictionaryMatch[] matches;
    private readonly StateDictionaryReplacement[] replacements;
    private readonly HashSet<string>? allowedKeys;

    public StateDictionaryOperations(
        string name,
        IEnumerable<StateDictionaryMatch>? matches = null,
        IEnumerable<StateDictionaryReplacement>? replacements = null,
        IEnumerable<string>? allowedKeys = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        this.matches = matches?.ToArray() ?? [new StateDictionaryMatch()];
        this.replacements = replacements?.ToArray() ?? [];
        this.allowedKeys = allowedKeys is null ? null : new HashSet<string>(allowedKeys, StringComparer.Ordinal);
    }

    public string Name { get; }

    public string? ApplyToKey(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (!matches.Any(match => match.Matches(key)))
        {
            return null;
        }

        foreach (var replacement in replacements)
        {
            key = key.Replace(replacement.Content, replacement.Replacement, StringComparison.Ordinal);
        }

        return allowedKeys is null || allowedKeys.Contains(key) ? key : null;
    }

    public static StateDictionaryOperations LtxLoraComfyRenaming { get; } = new(
        "LTXV_LORA_COMFY_PREFIX_MAP",
        replacements: [new StateDictionaryReplacement("diffusion_model.", "")]);
}
