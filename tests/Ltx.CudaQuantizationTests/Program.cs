using System.Text.Json;
using Ltx.Cuda;

if (args.Length != 1)
{
    Console.Error.WriteLine("usage: Ltx.CudaQuantizationTests <fixture-dir>");
    return 2;
}

using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Path.GetFullPath(args[0]), "manifest.json")));
var root = document.RootElement;
if (root.GetProperty("schema_version").GetInt32() != 1 ||
    root.GetProperty("fixture_revision").GetString() != "m4-cuda-quantization-lora-v1" ||
    root.GetProperty("oracle").GetProperty("framework_version").GetString() != "2.13.0+cu132" ||
    root.GetProperty("oracle").GetProperty("cuda_version").GetString() != "13.2")
{
    throw new InvalidDataException("M4 fixture versions do not match the pinned oracle.");
}

LtxCudaNative.ValidateAbi();
var fp32 = Tolerance(root.GetProperty("tolerances").GetProperty("fp32"));
var reduced = Tolerance(root.GetProperty("tolerances").GetProperty("bf16_fp8"));
var cases = root.GetProperty("cases");
var coveredKernels = new HashSet<string>(StringComparer.Ordinal);
var coveredBindings = new HashSet<string>(StringComparer.Ordinal);

var expectedKernels = Strings(root.GetProperty("native_kernel_inventory"));
var actualKernels = LtxCudaKernels.KernelInventory.ToArray();
AssertExact("native kernel inventory", actualKernels, expectedKernels);

var fused = cases.GetProperty("fused_add_round");
AssertNear("fused add/round", LtxCudaKernels.FusedAddRound(
    Floats(fused.GetProperty("delta")), Floats(fused.GetProperty("weight")), fused.GetProperty("seed").GetUInt32()),
    Floats(fused.GetProperty("expected")), reduced);
Cover(coveredKernels, "fused_add_round_kernel");

var na3d = cases.GetProperty("na3d");
var naShape = Ints(na3d.GetProperty("shape"));
var naKernel = Ints(na3d.GetProperty("kernel"));
AssertNear("3D neighborhood attention", LtxCudaKernels.NeighborhoodAttention3D(
    Floats(na3d.GetProperty("query")), Floats(na3d.GetProperty("key")), Floats(na3d.GetProperty("value")),
    naShape[0], naShape[1], naShape[2], naShape[3], naShape[4], naShape[5],
    naKernel[0], naKernel[1], naKernel[2]), Floats(na3d.GetProperty("expected")), fp32);
Cover(coveredKernels, "_na3d_kernel");

var swiglu = cases.GetProperty("swiglu");
var rows = swiglu.GetProperty("rows").GetInt32();
var inputDimension = swiglu.GetProperty("input_dim").GetInt32();
var outputDimension = swiglu.GetProperty("output_dim").GetInt32();
AssertNear("SwiGLU up multiply", LtxCudaKernels.SwiGluUpMultiply(
    Floats(swiglu.GetProperty("input")), Floats(swiglu.GetProperty("up_weight")), Floats(swiglu.GetProperty("gate")),
    rows, inputDimension, outputDimension), Floats(swiglu.GetProperty("up_mul_expected")), fp32);
AssertNear("SwiGLU fused gate/up", LtxCudaKernels.SwiGluGateUp(
    Floats(swiglu.GetProperty("input")), Floats(swiglu.GetProperty("gate_weight")), Floats(swiglu.GetProperty("up_weight")),
    rows, inputDimension, outputDimension), Floats(swiglu.GetProperty("gate_up_expected")), fp32);
Cover(coveredKernels, "_fused_up_mul_kernel", "_fused_gate_up_swiglu_kernel");

var gemm = cases.GetProperty("fp8_gemm");
rows = gemm.GetProperty("rows").GetInt32();
outputDimension = gemm.GetProperty("output_dim").GetInt32();
inputDimension = gemm.GetProperty("inner_dim").GetInt32();
var gemmLeft = Floats(gemm.GetProperty("left"));
var gemmRight = Floats(gemm.GetProperty("right"));
var gemmExpected = Floats(gemm.GetProperty("expected"));
AssertNear("SM89 FP8 GEMM", LtxCudaKernels.Fp8Gemm(
    gemmLeft, gemmRight, rows, outputDimension, inputDimension, Fp8GemmVariant.Sm89), gemmExpected, fp32);
AssertNear("SM90 FP8 GEMM", LtxCudaKernels.Fp8Gemm(
    gemmLeft, gemmRight, rows, outputDimension, inputDimension, Fp8GemmVariant.Sm90), gemmExpected, fp32);
AssertNear("SM90 FP8 GEMM bias", LtxCudaKernels.Fp8Gemm(
    gemmLeft, gemmRight, rows, outputDimension, inputDimension, Fp8GemmVariant.Sm90Bias,
    Floats(gemm.GetProperty("bias"))), Floats(gemm.GetProperty("bias_expected")), fp32);
Cover(coveredKernels, "gemm_fp8_kernel", "sm90_fp8_gemm_1d2d_impl", "sm90_fp8_gemm_1d2d_bias_impl");
Cover(coveredBindings, "fp8_gemm_nt_sm89", "fp8_gemm_nt_sm90");

var nvfp4 = cases.GetProperty("nvfp4");
var leftExpected = nvfp4.GetProperty("left");
var leftInput = Floats(nvfp4.GetProperty("left_input"));
var leftRows = leftExpected.GetProperty("rows").GetInt32();
var leftColumns = leftExpected.GetProperty("columns").GetInt32();
var leftScale = leftExpected.GetProperty("per_tensor_scale").GetSingle();
var leftOne = LtxCudaKernels.QuantizeNvFp4(leftInput, leftRows, leftColumns, variant: 1, perTensorScale: leftScale);
var leftTwo = LtxCudaKernels.QuantizeNvFp4(leftInput, leftRows, leftColumns, variant: 2, perTensorScale: leftScale);
AssertNvFp4("NVFP4 elementwise quantize", leftOne, leftExpected, fp32);
AssertNvFp4("NVFP4 tiled quantize", leftTwo, leftExpected, fp32);
AssertNear("NVFP4 dequantize", LtxCudaKernels.DequantizeNvFp4(leftTwo),
    Floats(leftExpected.GetProperty("dequantized")), reduced);
Cover(coveredKernels, "quantize_kernel", "quantize_tiled_kernel", "dequantize_kernel");
Cover(coveredBindings, "quantize_nvfp4", "dequantize_nvfp4");

var rightTensor = NvFp4(nvfp4.GetProperty("right"));
AssertNear("NVFP4 scaled matrix multiply", LtxCudaKernels.NvFp4ScaledMatrixMultiply(
    leftTwo, rightTensor, Floats(nvfp4.GetProperty("bias"))),
    Floats(nvfp4.GetProperty("scaled_mm_expected")), reduced);
Cover(coveredBindings, "scaled_mm_nvfp4");

var scalars = cases.GetProperty("scalars");
AssertNear("scalar multiply", [LtxCudaKernels.MultiplyScalars(
    scalars.GetProperty("left").GetSingle(), scalars.GetProperty("right").GetSingle())],
    [scalars.GetProperty("product").GetSingle()], fp32);
AssertNear("amax scale", [LtxCudaKernels.AmaxScale(
    Floats(scalars.GetProperty("amax_input")), scalars.GetProperty("divisor").GetSingle())],
    [scalars.GetProperty("amax_expected").GetSingle()], fp32);
Cover(coveredKernels, "mul_scalars_kernel", "amax_scale_kernel");
Cover(coveredBindings, "mul_scalars", "amax_scale");

if (LtxCudaKernels.GemmAlphaMode != 1 || !LtxCudaKernels.ProbeGemmSupport(2, 8, 32))
    throw new InvalidDataException("NVFP4 GEMM diagnostic bindings failed.");
LtxCudaKernels.ConfigureGemm(alphaOnDevice: true, autotune: false);
Cover(coveredBindings, "gemm_alpha_mode", "set_gemm_alpha_on_device", "set_gemm_autotune", "probe_gemm_support");

var fp6 = cases.GetProperty("fp6");
rows = fp6.GetProperty("rows").GetInt32();
var columns = fp6.GetProperty("columns").GetInt32();
var packedFp6 = LtxCudaKernels.PackFp6(Bytes(fp6.GetProperty("input")), rows, columns);
AssertExact("FP6 pack", packedFp6, Bytes(fp6.GetProperty("packed")));
AssertExact("FP6 unpack", LtxCudaKernels.UnpackFp6(packedFp6, rows, columns), Bytes(fp6.GetProperty("unpacked")));
Cover(coveredKernels, "fp6_pack_kernel", "fp6_unpack_kernel");
Cover(coveredBindings, "fp6_pack", "fp6_unpack");

var rope = cases.GetProperty("rms_rope");
rows = rope.GetProperty("rows").GetInt32();
var hiddenDimension = rope.GetProperty("hidden_dim").GetInt32();
AssertNear("RMS norm adjacent RoPE", LtxCudaKernels.RmsNormRope(
    Floats(rope.GetProperty("input")), Floats(rope.GetProperty("cosine")), Floats(rope.GetProperty("sine")),
    rows, hiddenDimension, Floats(rope.GetProperty("weights"))), Floats(rope.GetProperty("expected")), fp32);
Cover(coveredKernels, "norm_rope_cvt_kernel");
Cover(coveredBindings, "rms_norm_rope");

var split = cases.GetProperty("rms_split_rope");
rows = split.GetProperty("rows").GetInt32();
var heads = split.GetProperty("heads").GetInt32();
var headDimension = split.GetProperty("head_dim").GetInt32();
AssertNear("RMS norm split RoPE", LtxCudaKernels.RmsNormSplitRope(
    Floats(split.GetProperty("input")), Floats(split.GetProperty("weights")),
    Floats(split.GetProperty("cosine")), Floats(split.GetProperty("sine")),
    rows, heads, headDimension), Floats(split.GetProperty("expected")), fp32);
Cover(coveredKernels, "_rms_norm_split_rope_kernel");
Cover(coveredBindings, "rms_norm_split_rope");

var rowwise = cases.GetProperty("rowwise_int8");
rows = rowwise.GetProperty("rows").GetInt32();
columns = rowwise.GetProperty("columns").GetInt32();
var rowwiseActual = LtxCudaKernels.QuantizeRowwiseInt8(Floats(rowwise.GetProperty("input")), rows, columns);
AssertExact("rowwise INT8 values", rowwiseActual.Values, SBytes(rowwise.GetProperty("values")));
AssertNear("rowwise INT8 scales", rowwiseActual.Scales, Floats(rowwise.GetProperty("scales")), fp32);
Cover(coveredKernels, "_kernel");

var blockwise = cases.GetProperty("blockwise");
rows = blockwise.GetProperty("rows").GetInt32();
columns = blockwise.GetProperty("columns").GetInt32();
var blockSize = blockwise.GetProperty("block_size").GetInt32();
var blockInput = Floats(blockwise.GetProperty("input"));
var plain = LtxCudaKernels.QuantizeBlockwiseFp8(blockInput, rows, columns, blockSize);
AssertBlockwise("blockwise FP8", plain, blockwise.GetProperty("plain"), fp32);
AssertNear("blockwise FP8 dequantize", LtxCudaKernels.DequantizeBlockwiseFp8(plain),
    Floats(blockwise.GetProperty("plain").GetProperty("dequantized")), reduced);
var gelu = LtxCudaKernels.QuantizeBlockwiseFp8(blockInput, rows, columns, blockSize, useGelu: true);
AssertBlockwise("blockwise FP8 GELU", gelu, blockwise.GetProperty("gelu"), fp32);
var norm = LtxCudaKernels.QuantizeBlockwiseFp8WithNorm(
    blockInput, Floats(blockwise.GetProperty("norm_scale")), Floats(blockwise.GetProperty("norm_shift")),
    rows, columns, blockSize);
AssertBlockwise("blockwise FP8 AdaNorm", norm, blockwise.GetProperty("norm"), fp32);
Cover(coveredKernels, "_quantize", "_block_quant_kernel", "_gelu", "_block_quant_norm_kernel", "_blockwise_dequantize_kernel");

var fma = cases.GetProperty("quant_rms_fma");
rows = fma.GetProperty("rows").GetInt32();
columns = fma.GetProperty("columns").GetInt32();
blockSize = fma.GetProperty("block_size").GetInt32();
var fmaActual = LtxCudaKernels.QuantizedRmsFma(
    Floats(fma.GetProperty("x")), Floats(fma.GetProperty("y")), Floats(fma.GetProperty("z")),
    rows, columns, blockSize);
AssertNear("quantized RMS FMA residual", fmaActual.Residual, Floats(fma.GetProperty("residual")), fp32);
AssertBlockwise("quantized RMS FMA", fmaActual.Quantized, fma.GetProperty("quantized"), fp32);
Cover(coveredKernels, "_quant_rms_sum_mult_kernel");

var gated = cases.GetProperty("gated_attention");
rows = gated.GetProperty("rows").GetInt32();
heads = gated.GetProperty("heads").GetInt32();
headDimension = gated.GetProperty("head_dim").GetInt32();
AssertNear("gated attention", LtxCudaKernels.GatedAttention(
    Floats(gated.GetProperty("input")), Floats(gated.GetProperty("gate_logits")),
    rows, heads, headDimension), Floats(gated.GetProperty("expected")), fp32);
Cover(coveredKernels, "_gated_attention_kernel");

var lora = cases.GetProperty("lora");
var loraFusion = LtxCudaKernels.FuseNvFp4Lora(
    rightTensor, Floats(lora.GetProperty("factor_a")), Floats(lora.GetProperty("factor_b")),
    lora.GetProperty("rank").GetInt32(), lora.GetProperty("strength").GetSingle());
AssertNear("NVFP4 LoRA fuse", LtxCudaKernels.DequantizeNvFp4(loraFusion.Fused),
    Floats(lora.GetProperty("fused_dequantized")), reduced);
AssertNear("NVFP4 LoRA unfuse", LtxCudaKernels.DequantizeNvFp4(loraFusion.Unfuse()),
    Floats(lora.GetProperty("unfused_dequantized")), reduced);

AssertExact("covered native kernels", coveredKernels.Order(StringComparer.Ordinal).ToArray(),
    expectedKernels.Order(StringComparer.Ordinal).ToArray());
var expectedBindings = Strings(root.GetProperty("native_binding_inventory"));
AssertExact("covered native bindings", coveredBindings.Order(StringComparer.Ordinal).ToArray(),
    expectedBindings.Order(StringComparer.Ordinal).ToArray());

const int quantizationFixtures = 4;
const int loraFixtures = 2;
var total = expectedKernels.Length + expectedBindings.Length + quantizationFixtures + loraFixtures;
Console.WriteLine(
    $"M4 CUDA/quantization/LoRA fixtures: {total} passed, 0 failed; " +
    $"kernels={expectedKernels.Length}, bindings={expectedBindings.Length}, quantization={quantizationFixtures}, lora={loraFixtures}; " +
    $"FP32 rtol={fp32.Relative:g}, atol={fp32.Absolute:g}; reduced rtol={reduced.Relative:g}, atol={reduced.Absolute:g}");
return 0;

static void AssertNvFp4(string name, NvFp4Tensor actual, JsonElement expected, NumericTolerance tolerance)
{
    AssertExact(name + " packed", actual.Packed, Bytes(expected.GetProperty("packed")));
    AssertExact(name + " block scales", actual.BlockScales, Bytes(expected.GetProperty("block_scales")));
    AssertNear(name + " per-tensor scale", [actual.PerTensorScale],
        [expected.GetProperty("per_tensor_scale").GetSingle()], tolerance);
}

static void AssertBlockwise(string name, BlockwiseFp8Tensor actual, JsonElement expected, NumericTolerance tolerance)
{
    AssertExact(name + " values", actual.Values, Bytes(expected.GetProperty("values")));
    AssertNear(name + " scales", actual.Scales, Floats(expected.GetProperty("scales")), tolerance);
}

static NvFp4Tensor NvFp4(JsonElement element) => new(
    Bytes(element.GetProperty("packed")),
    Bytes(element.GetProperty("block_scales")),
    element.GetProperty("rows").GetInt32(),
    element.GetProperty("columns").GetInt32(),
    element.GetProperty("per_tensor_scale").GetSingle(),
    element.GetProperty("high_nibble_first").GetBoolean());

static void Cover(ISet<string> destination, params string[] names)
{
    foreach (var name in names)
    {
        if (!destination.Add(name)) throw new InvalidDataException($"Coverage '{name}' was recorded twice.");
    }
}

static void AssertNear(string name, IReadOnlyList<float> actual, IReadOnlyList<float> expected, NumericTolerance tolerance)
{
    if (actual.Count != expected.Count) throw new InvalidDataException($"{name}: length mismatch.");
    for (var index = 0; index < actual.Count; index++)
    {
        var allowed = tolerance.Absolute + tolerance.Relative * MathF.Abs(expected[index]);
        var error = MathF.Abs(actual[index] - expected[index]);
        if (error > allowed)
            throw new InvalidDataException(
                $"{name}[{index}] mismatch: actual={actual[index]:g9}, expected={expected[index]:g9}, allowed={allowed:g9}.");
    }
}

static void AssertExact<T>(string name, IReadOnlyList<T> actual, IReadOnlyList<T> expected)
{
    if (actual.Count != expected.Count || !actual.SequenceEqual(expected))
        throw new InvalidDataException($"{name}: exact sequence mismatch.");
}

static NumericTolerance Tolerance(JsonElement element) =>
    new(element.GetProperty("rtol").GetSingle(), element.GetProperty("atol").GetSingle());

static float[] Floats(JsonElement element) => element.EnumerateArray().Select(value => value.GetSingle()).ToArray();
static byte[] Bytes(JsonElement element) => element.EnumerateArray().Select(value => value.GetByte()).ToArray();
static sbyte[] SBytes(JsonElement element) => element.EnumerateArray().Select(value => value.GetSByte()).ToArray();
static int[] Ints(JsonElement element) => element.EnumerateArray().Select(value => value.GetInt32()).ToArray();
static string[] Strings(JsonElement element) => element.EnumerateArray().Select(value => value.GetString()!).ToArray();

internal readonly record struct NumericTolerance(float Relative, float Absolute);
