using System.Text.Json.Serialization;

namespace Ltx.StorageTests;

internal sealed record FixtureManifest(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("fixture_revision")] string FixtureRevision,
    [property: JsonPropertyName("oracle")] OracleInfo Oracle,
    [property: JsonPropertyName("tolerances")] IReadOnlyDictionary<string, Tolerance> Tolerances,
    [property: JsonPropertyName("metadata")] IReadOnlyDictionary<string, string> Metadata,
    [property: JsonPropertyName("mixed_tensors")] IReadOnlyDictionary<string, TensorCase> MixedTensors,
    [property: JsonPropertyName("checkpoint")] CheckpointFixture Checkpoint,
    [property: JsonPropertyName("lora")] LoraFixture Lora,
    [property: JsonPropertyName("file_sha256")] IReadOnlyDictionary<string, string> FileSha256);

internal sealed record OracleInfo(
    [property: JsonPropertyName("framework_version")] string FrameworkVersion,
    [property: JsonPropertyName("cuda_version")] string CudaVersion,
    [property: JsonPropertyName("safetensors_version")] string SafetensorsVersion,
    [property: JsonPropertyName("seed")] int Seed);

internal sealed record Tolerance(
    [property: JsonPropertyName("rtol")] float Relative,
    [property: JsonPropertyName("atol")] float Absolute);

internal sealed record TensorCase(
    [property: JsonPropertyName("dtype")] string DType,
    [property: JsonPropertyName("shape")] long[] Shape,
    [property: JsonPropertyName("values")] float[] Values);

internal sealed record CheckpointFixture(
    [property: JsonPropertyName("paths")] string[] Paths,
    [property: JsonPropertyName("expected")] IReadOnlyDictionary<string, TensorCase> Expected);

internal sealed record LoraFixture(
    [property: JsonPropertyName("path")] string Path,
    [property: JsonPropertyName("metadata")] IReadOnlyDictionary<string, string> Metadata,
    [property: JsonPropertyName("strength")] double Strength,
    [property: JsonPropertyName("mapped_keys")] string[] MappedKeys,
    [property: JsonPropertyName("expected_fused")] IReadOnlyDictionary<string, TensorCase> ExpectedFused,
    [property: JsonPropertyName("expected_unfused")] IReadOnlyDictionary<string, TensorCase> ExpectedUnfused);
