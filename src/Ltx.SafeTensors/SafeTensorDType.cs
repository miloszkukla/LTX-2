namespace Ltx.SafeTensors;

public enum SafeTensorDType
{
    Bool,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float4,
    Float6E2M3,
    Float6E3M2,
    Float8E4M3,
    Float8E4M3Fnuz,
    Float8E5M2,
    Float8E5M2Fnuz,
    Float8E8M0,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Complex64,
}

public static class SafeTensorDTypeExtensions
{
    public static int BitsPerElement(this SafeTensorDType dtype) => dtype switch
    {
        SafeTensorDType.Float4 => 4,
        SafeTensorDType.Float6E2M3 or SafeTensorDType.Float6E3M2 => 6,
        SafeTensorDType.Bool or SafeTensorDType.UInt8 or SafeTensorDType.Int8 or
            SafeTensorDType.Float8E4M3 or SafeTensorDType.Float8E4M3Fnuz or
            SafeTensorDType.Float8E5M2 or SafeTensorDType.Float8E5M2Fnuz or
            SafeTensorDType.Float8E8M0 => 8,
        SafeTensorDType.UInt16 or SafeTensorDType.Int16 or SafeTensorDType.Float16 or
            SafeTensorDType.BFloat16 => 16,
        SafeTensorDType.UInt32 or SafeTensorDType.Int32 or SafeTensorDType.Float32 => 32,
        SafeTensorDType.UInt64 or SafeTensorDType.Int64 or SafeTensorDType.Float64 or
            SafeTensorDType.Complex64 => 64,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
    };

    public static string ToFormatCode(this SafeTensorDType dtype) => dtype switch
    {
        SafeTensorDType.Bool => "BOOL",
        SafeTensorDType.UInt8 => "U8",
        SafeTensorDType.Int8 => "I8",
        SafeTensorDType.UInt16 => "U16",
        SafeTensorDType.Int16 => "I16",
        SafeTensorDType.UInt32 => "U32",
        SafeTensorDType.Int32 => "I32",
        SafeTensorDType.UInt64 => "U64",
        SafeTensorDType.Int64 => "I64",
        SafeTensorDType.Float4 => "F4",
        SafeTensorDType.Float6E2M3 => "F6_E2M3",
        SafeTensorDType.Float6E3M2 => "F6_E3M2",
        SafeTensorDType.Float8E4M3 => "F8_E4M3",
        SafeTensorDType.Float8E4M3Fnuz => "F8_E4M3FNUZ",
        SafeTensorDType.Float8E5M2 => "F8_E5M2",
        SafeTensorDType.Float8E5M2Fnuz => "F8_E5M2FNUZ",
        SafeTensorDType.Float8E8M0 => "F8_E8M0",
        SafeTensorDType.Float16 => "F16",
        SafeTensorDType.BFloat16 => "BF16",
        SafeTensorDType.Float32 => "F32",
        SafeTensorDType.Float64 => "F64",
        SafeTensorDType.Complex64 => "C64",
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
    };

    public static SafeTensorDType ParseFormatCode(string code) => code switch
    {
        "BOOL" => SafeTensorDType.Bool,
        "U8" => SafeTensorDType.UInt8,
        "I8" => SafeTensorDType.Int8,
        "U16" => SafeTensorDType.UInt16,
        "I16" => SafeTensorDType.Int16,
        "U32" => SafeTensorDType.UInt32,
        "I32" => SafeTensorDType.Int32,
        "U64" => SafeTensorDType.UInt64,
        "I64" => SafeTensorDType.Int64,
        "F4" => SafeTensorDType.Float4,
        "F6_E2M3" => SafeTensorDType.Float6E2M3,
        "F6_E3M2" => SafeTensorDType.Float6E3M2,
        "F8_E4M3" => SafeTensorDType.Float8E4M3,
        "F8_E4M3FNUZ" => SafeTensorDType.Float8E4M3Fnuz,
        "F8_E5M2" => SafeTensorDType.Float8E5M2,
        "F8_E5M2FNUZ" => SafeTensorDType.Float8E5M2Fnuz,
        "F8_E8M0" => SafeTensorDType.Float8E8M0,
        "F16" => SafeTensorDType.Float16,
        "BF16" => SafeTensorDType.BFloat16,
        "F32" => SafeTensorDType.Float32,
        "F64" => SafeTensorDType.Float64,
        "C64" => SafeTensorDType.Complex64,
        _ => throw new InvalidDataException($"Unsupported safetensors dtype '{code}'."),
    };
}
