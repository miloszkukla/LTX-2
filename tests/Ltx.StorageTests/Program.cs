using System.Text.Json;
using Ltx.Core;
using Ltx.SafeTensors;
using Ltx.StorageTests;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: Ltx.StorageTests <fixture-dir> <output-dir> <libLibTorchSharp.so>");
    return 2;
}

var fixtureDirectory = Path.GetFullPath(args[0]);
var outputDirectory = Path.GetFullPath(args[1]);
Directory.CreateDirectory(outputDirectory);
var manifest = JsonSerializer.Deserialize<FixtureManifest>(
    File.ReadAllText(Path.Combine(fixtureDirectory, "manifest.json")))
    ?? throw new InvalidDataException("M2 fixture manifest is empty.");
if (manifest.SchemaVersion != 1 || manifest.FixtureRevision != "m2-storage-model-loading-v1" ||
    manifest.Oracle.FrameworkVersion != "2.13.0+cu132" || manifest.Oracle.CudaVersion != AbiContract.CudaDisplayVersion ||
    manifest.Oracle.SafetensorsVersion != "0.6.2")
{
    throw new InvalidDataException("M2 fixture versions do not match the pinned storage oracle.");
}

TorchSharpRuntime.Initialize(args[2]);
var passed = 0;
var mixedPath = Path.Combine(fixtureDirectory, "storage-mixed.safetensors");
var mixed = SafeTensorFile.Load(mixedPath);
AssertMetadata("mixed metadata", mixed.Metadata, manifest.Metadata);
passed++;

AssertKeys("mixed keys", mixed.Tensors.Keys, manifest.MixedTensors.Keys);
foreach (var (name, expected) in manifest.MixedTensors)
{
    var actual = mixed.Tensors[name];
    if (!actual.Shape.SequenceEqual(expected.Shape))
    {
        throw new InvalidDataException($"{name}: shape mismatch.");
    }
}
passed++;

var storageRoundtripPath = Path.Combine(outputDirectory, "storage-roundtrip.safetensors");
mixed.Save(storageRoundtripPath);
AssertSameFile("raw safetensors round trip", mixed, SafeTensorFile.Load(storageRoundtripPath));
passed++;

var reconstructed = new Dictionary<string, SafeTensor>(StringComparer.Ordinal);
foreach (var (name, expected) in manifest.MixedTensors)
{
    using var tensor = mixed.Tensors[name].ToTorchTensor();
    AssertTensor(name, tensor, expected, ToleranceFor(expected, manifest));
    reconstructed.Add(name, SafeTensor.FromTorchTensor(tensor));
}
passed++;

var torchRoundtrip = new SafeTensorFile(reconstructed, mixed.Metadata);
var torchRoundtripPath = Path.Combine(outputDirectory, "torch-roundtrip.safetensors");
torchRoundtrip.Save(torchRoundtripPath);
AssertSameFile("TorchSharp safetensors round trip", mixed, SafeTensorFile.Load(torchRoundtripPath));
passed++;

var loader = new SafetensorsCheckpointLoader();
var checkpointPaths = manifest.Checkpoint.Paths.Select(path => Path.Combine(fixtureDirectory, path)).ToArray();
var checkpointMetadata = loader.ReadMetadata(checkpointPaths[0]);
AssertMetadata("checkpoint metadata", checkpointMetadata.Raw, manifest.Metadata);
if (checkpointMetadata.ModelVersion != "2.5.0" || checkpointMetadata.ModelConfig is not { } config ||
    config.GetProperty("model").GetProperty("hidden_size").GetInt32() != 4)
{
    throw new InvalidDataException("Checkpoint JSON metadata parsing failed.");
}
passed++;

var checkpointOperations = new StateDictionaryOperations(
    "M2_CHECKPOINT_MAP",
    matches: [new StateDictionaryMatch(Prefix: "diffusion_model.")],
    replacements: [new StateDictionaryReplacement("diffusion_model.", "")]);
using var checkpoint = loader.Load(checkpointPaths, checkpointOperations);
AssertKeys("checkpoint keys", checkpoint.Tensors.Keys, manifest.Checkpoint.Expected.Keys);
foreach (var (name, expected) in manifest.Checkpoint.Expected)
{
    AssertTensor(name, checkpoint.Tensors[name], expected, ToleranceFor(expected, manifest));
}
passed++;

var loraPath = Path.Combine(fixtureDirectory, manifest.Lora.Path);
var loraFile = SafeTensorFile.Load(loraPath);
AssertMetadata("LoRA metadata", loraFile.Metadata, manifest.Lora.Metadata);
var loraRoundtripPath = Path.Combine(outputDirectory, "lora-roundtrip.safetensors");
loraFile.Save(loraRoundtripPath);
AssertSameFile("LoRA round trip", loraFile, SafeTensorFile.Load(loraRoundtripPath));
passed++;

using var lora = loader.Load(loraPath, StateDictionaryOperations.LtxLoraComfyRenaming);
AssertKeys("mapped LoRA keys", lora.Tensors.Keys, manifest.Lora.MappedKeys);
passed++;

var adapter = new LoraAdapter(lora, manifest.Lora.Strength);
using var fused = LoraFusion.Fuse(checkpoint, [adapter]);
foreach (var (name, expected) in manifest.Lora.ExpectedFused)
{
    AssertTensor(name, fused.Tensors[name], expected, ToleranceFor(expected, manifest));
}
passed += 2;

using var unfused = LoraFusion.Unfuse(fused, [adapter]);
foreach (var (name, expected) in manifest.Lora.ExpectedUnfused)
{
    AssertTensor(name, unfused.Tensors[name], expected, ToleranceFor(expected, manifest));
}
passed += 2;

Console.WriteLine(
    $"M2 storage fixtures: {passed} passed, 0 failed; " +
    $"FP32 rtol={manifest.Tolerances["fp32"].Relative:g}, atol={manifest.Tolerances["fp32"].Absolute:g}; " +
    $"BF16 rtol={manifest.Tolerances["bf16"].Relative:g}, atol={manifest.Tolerances["bf16"].Absolute:g}; " +
    $"revision={manifest.FixtureRevision}");
return 0;

static Tolerance ToleranceFor(TensorCase tensorCase, FixtureManifest manifest) =>
    tensorCase.DType == "bfloat16" || tensorCase.DType == "float16"
        ? manifest.Tolerances["bf16"]
        : manifest.Tolerances["fp32"];

static void AssertTensor(string name, Tensor actual, TensorCase expected, Tolerance tolerance)
{
    if (!actual.shape.SequenceEqual(expected.Shape))
    {
        throw new InvalidDataException($"{name}: tensor shape mismatch.");
    }

    if (expected.Values.Length == 0)
    {
        if (actual.numel() != 0)
        {
            throw new InvalidDataException($"{name}: expected an empty tensor.");
        }

        return;
    }

    using var floatTensor = actual.to_type(ScalarType.Float32);
    using var cpu = floatTensor.cpu();
    var values = cpu.data<float>().ToArray();
    if (values.Length != expected.Values.Length)
    {
        throw new InvalidDataException($"{name}: tensor length mismatch.");
    }

    for (var index = 0; index < values.Length; index++)
    {
        var allowed = tolerance.Absolute + tolerance.Relative * MathF.Abs(expected.Values[index]);
        var error = MathF.Abs(values[index] - expected.Values[index]);
        if (error > allowed)
        {
            throw new InvalidDataException(
                $"{name}[{index}] mismatch: actual={values[index]:g9}, expected={expected.Values[index]:g9}, allowed={allowed:g9}.");
        }
    }
}

static void AssertKeys(string name, IEnumerable<string> actual, IEnumerable<string> expected)
{
    var actualKeys = actual.Order(StringComparer.Ordinal).ToArray();
    var expectedKeys = expected.Order(StringComparer.Ordinal).ToArray();
    if (!actualKeys.SequenceEqual(expectedKeys, StringComparer.Ordinal))
    {
        throw new InvalidDataException(
            $"{name}: actual=[{string.Join(",", actualKeys)}], expected=[{string.Join(",", expectedKeys)}].");
    }
}

static void AssertMetadata(
    string name,
    IReadOnlyDictionary<string, string> actual,
    IReadOnlyDictionary<string, string> expected)
{
    if (actual.Count != expected.Count || expected.Any(pair => !actual.TryGetValue(pair.Key, out var value) || value != pair.Value))
    {
        throw new InvalidDataException($"{name}: metadata mismatch.");
    }
}

static void AssertSameFile(string name, SafeTensorFile expected, SafeTensorFile actual)
{
    AssertMetadata(name, actual.Metadata, expected.Metadata);
    AssertKeys(name, actual.Tensors.Keys, expected.Tensors.Keys);
    foreach (var (key, expectedTensor) in expected.Tensors)
    {
        var actualTensor = actual.Tensors[key];
        if (actualTensor.DType != expectedTensor.DType || !actualTensor.Shape.SequenceEqual(expectedTensor.Shape) ||
            !actualTensor.Data.Span.SequenceEqual(expectedTensor.Data.Span))
        {
            throw new InvalidDataException($"{name}: tensor '{key}' changed.");
        }
    }
}
