namespace Ltx.Cuda;

public sealed class LtxCudaException(string operation, int status)
    : Exception($"{operation} failed with CUDA status {status}.")
{
    public int Status { get; } = status;
}
