using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Core;

public enum OffloadMode
{
    None,
    Cpu,
    Disk,
}

public sealed class OffloadedLinear : IDisposable
{
    private readonly Device targetDevice;
    private readonly ScalarType computeType;
    private readonly string? checkpointPath;
    private readonly string weightKey;
    private readonly string biasKey;
    private Tensor? residentWeight;
    private Tensor? residentBias;
    private bool disposed;

    public OffloadedLinear(
        Tensor weight,
        Tensor bias,
        OffloadMode mode,
        Device targetDevice,
        ScalarType computeType)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(bias);
        if (mode == OffloadMode.Disk)
        {
            throw new ArgumentException("Use the checkpoint constructor for disk offload.", nameof(mode));
        }

        Mode = mode;
        this.targetDevice = targetDevice;
        this.computeType = computeType;
        weightKey = string.Empty;
        biasKey = string.Empty;
        var residentDevice = mode == OffloadMode.None ? targetDevice : CPU;
        residentWeight = weight.to(computeType, residentDevice, copy: true);
        residentBias = bias.to(computeType, residentDevice, copy: true);
    }

    public OffloadedLinear(
        string checkpointPath,
        string weightKey,
        string biasKey,
        Device targetDevice,
        ScalarType computeType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(weightKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(biasKey);
        Mode = OffloadMode.Disk;
        this.targetDevice = targetDevice;
        this.computeType = computeType;
        this.checkpointPath = Path.GetFullPath(checkpointPath);
        this.weightKey = weightKey;
        this.biasKey = biasKey;
    }

    public OffloadMode Mode { get; }

    public int DiskLoads { get; private set; }

    public string Residency => Mode switch
    {
        OffloadMode.None => "cuda",
        OffloadMode.Cpu => "cpu",
        OffloadMode.Disk => "disk",
        _ => throw new InvalidOperationException(),
    };

    public Tensor Forward(Tensor input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        StateDictionary? checkpoint = null;
        using var scope = NewDisposeScope();
        try
        {
            Tensor sourceWeight;
            Tensor sourceBias;
            if (Mode == OffloadMode.Disk)
            {
                checkpoint = new SafetensorsCheckpointLoader().Load(checkpointPath!);
                if (!checkpoint.TryGetTensor(weightKey, out sourceWeight) ||
                    !checkpoint.TryGetTensor(biasKey, out sourceBias))
                {
                    throw new InvalidDataException("Offload checkpoint does not contain the requested linear tensors.");
                }
                DiskLoads++;
            }
            else
            {
                sourceWeight = residentWeight!;
                sourceBias = residentBias!;
            }

            var weight = Mode == OffloadMode.None
                ? sourceWeight
                : sourceWeight.to(computeType, targetDevice, non_blocking: false);
            var bias = Mode == OffloadMode.None
                ? sourceBias
                : sourceBias.to(computeType, targetDevice, non_blocking: false);
            var output = input.matmul(weight.transpose(0, 1)).add(bias);
            return output.MoveToOuterDisposeScope();
        }
        finally
        {
            checkpoint?.Dispose();
        }
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        residentWeight?.Dispose();
        residentBias?.Dispose();
        residentWeight = null;
        residentBias = null;
    }
}
