using Ltx.Core;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

if (args.Length != 4)
{
    Console.Error.WriteLine(
        "usage: Ltx.UpsamplerTests <checkpoint> <spatial-upscaler> <oracle.safetensors> <libLibTorchSharp.so>");
    return 2;
}

TorchSharpRuntime.Initialize(args[3]);
InitializeDeviceType(DeviceType.CUDA);
if (!cuda.is_available() || cuda.device_count() != 1)
{
    throw new InvalidOperationException("Upsampler test requires exactly one CUDA device.");
}

using var scope = NewDisposeScope();
using var store = new CheckpointTensorStore([args[0]]);
using var upsampler = new CheckpointLatentUpsampler(store, args[1], CUDA);
var oracle = SafeTensorIndex.Open(args[2]);
if (oracle.Metadata.GetValueOrDefault("fixture_revision") != "m7c-learned-spatial-upscaler-v1")
{
    throw new InvalidDataException("Learned spatial upsampler oracle revision is invalid.");
}
using var input = oracle.LoadTorchTensor("input", CUDA);
using var output = upsampler.Upsample(input);
using var expected = oracle.LoadTorchTensor("expected", CUDA);
if (!output.shape.SequenceEqual([1L, 128L, 2L, 4L, 6L]) || !output.allclose(expected, 0.02, 0.005) ||
    output.isnan().any().item<bool>() || output.isinf().any().item<bool>())
{
    throw new InvalidDataException("Learned spatial upsampler output is invalid.");
}
Console.WriteLine("C# learned spatial upsampler: pass");
return 0;
