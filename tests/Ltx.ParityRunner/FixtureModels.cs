using System.Text.Json.Serialization;

namespace Ltx.ParityRunner;

internal sealed record FixtureDocument(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("fixture_revision")] string FixtureRevision,
    [property: JsonPropertyName("oracle")] OracleInfo Oracle,
    [property: JsonPropertyName("tolerance")] Tolerance Tolerance,
    [property: JsonPropertyName("torchsharp_case")] TorchSharpCase TorchSharpCase,
    [property: JsonPropertyName("native_case")] NativeCase NativeCase);

internal sealed record OracleInfo(
    [property: JsonPropertyName("framework_version")] string FrameworkVersion,
    [property: JsonPropertyName("cuda_version")] string CudaVersion,
    [property: JsonPropertyName("seed")] int Seed);

internal sealed record Tolerance(
    [property: JsonPropertyName("rtol")] float Relative,
    [property: JsonPropertyName("atol")] float Absolute);

internal sealed record TorchSharpCase(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("input")] float[] Input,
    [property: JsonPropertyName("input_shape")] long[] InputShape,
    [property: JsonPropertyName("weight")] float[] Weight,
    [property: JsonPropertyName("weight_shape")] long[] WeightShape,
    [property: JsonPropertyName("bias")] float[] Bias,
    [property: JsonPropertyName("expected")] float[] Expected,
    [property: JsonPropertyName("expected_shape")] long[] ExpectedShape);

internal sealed record NativeCase(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("input")] float[] Input,
    [property: JsonPropertyName("scale")] float Scale,
    [property: JsonPropertyName("bias")] float Bias,
    [property: JsonPropertyName("expected")] float[] Expected);
