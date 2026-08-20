namespace Ltx.Core;

public static class AbiContract
{
    public const int AbiMajor = 1;
    public const int AbiMinor = 0;
    public const string PyTorchVersion = "2.13.0";
    public const int CudaVersion = 13020;
    public const string CudaDisplayVersion = "13.2";
    public const int Cxx11Abi = 1;
    public const string TorchSharpManagedVersion = "0.107.0";
    public const string TorchSharpSourceRevision = "8f4def03b641b6753f18076aa5438f8eaaef2d30";

    public static uint PackedVersion => ((uint)AbiMajor << 16) | (uint)AbiMinor;
}
