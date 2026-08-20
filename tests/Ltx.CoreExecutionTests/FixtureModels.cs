using System.Text.Json.Serialization;

namespace Ltx.CoreExecutionTests;

internal sealed record FixtureManifest(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("fixture_revision")] string FixtureRevision,
    [property: JsonPropertyName("oracle")] OracleInfo Oracle,
    [property: JsonPropertyName("tolerances")] IReadOnlyDictionary<string, Tolerance> Tolerances,
    [property: JsonPropertyName("cases")] IReadOnlyDictionary<string, FixtureCase> Cases,
    [property: JsonPropertyName("file_sha256")] IReadOnlyDictionary<string, string> FileSha256);

internal sealed record OracleInfo(
    [property: JsonPropertyName("framework_version")] string FrameworkVersion,
    [property: JsonPropertyName("cuda_version")] string CudaVersion,
    [property: JsonPropertyName("safetensors_version")] string SafetensorsVersion,
    [property: JsonPropertyName("seed")] int Seed,
    [property: JsonPropertyName("device")] string Device);

internal sealed record Tolerance(
    [property: JsonPropertyName("rtol")] float Relative,
    [property: JsonPropertyName("atol")] float Absolute);

internal sealed record FixtureCase(
    [property: JsonPropertyName("suite")] string Suite,
    [property: JsonPropertyName("tensors")] IReadOnlyDictionary<string, TensorData> Tensors,
    [property: JsonPropertyName("integers")] IReadOnlyDictionary<string, long> Integers,
    [property: JsonPropertyName("numbers")] IReadOnlyDictionary<string, double> Numbers,
    [property: JsonPropertyName("flags")] IReadOnlyDictionary<string, bool> Flags,
    [property: JsonPropertyName("expected")] IReadOnlyDictionary<string, TensorData> Expected);

internal sealed record TensorData(
    [property: JsonPropertyName("shape")] long[] Shape,
    [property: JsonPropertyName("values")] float[] Values);
