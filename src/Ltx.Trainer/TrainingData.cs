using System.Globalization;
using Ltx.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Ltx.Trainer;

public sealed record TrainingPreprocessInput(
    Tensor Video,
    Tensor VideoEncoderWeight,
    Tensor VideoEncoderBias,
    Tensor CaptionFeatures,
    Tensor ConnectorWeight,
    Tensor ConnectorBias,
    Tensor PromptAttentionMask,
    double FramesPerSecond);

public sealed record TrainingPreprocessResult(string VideoLatentsPath, string ConditionsPath);

public static class TrainingPreprocessor
{
    public const string Schema = "ltx-csharp-precomputed-v1";

    public static TrainingPreprocessResult Process(
        string outputRoot,
        string sampleId,
        TrainingPreprocessInput input,
        bool overwrite = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(sampleId);
        ArgumentNullException.ThrowIfNull(input);
        Validate(input);

        var root = Path.GetFullPath(outputRoot);
        var videoPath = Path.Combine(root, "video_latents", sampleId + ".safetensors");
        var conditionsPath = Path.Combine(root, "conditions", sampleId + ".safetensors");
        if (!overwrite && (File.Exists(videoPath) || File.Exists(conditionsPath)))
        {
            throw new IOException($"Precomputed sample '{sampleId}' already exists.");
        }

        using var scope = NewDisposeScope();
        var channels = input.Video.shape[0];
        var frames = input.Video.shape[1];
        var height = input.Video.shape[2];
        var width = input.Video.shape[3];
        var latentChannels = input.VideoEncoderWeight.shape[0];
        var pixels = input.Video.permute(1, 2, 3, 0).reshape(-1, channels);
        var encoded = pixels.matmul(input.VideoEncoderWeight.transpose(0, 1))
            .add(input.VideoEncoderBias)
            .reshape(frames, height, width, latentChannels)
            .permute(3, 0, 1, 2)
            .contiguous();
        var promptEmbeds = input.CaptionFeatures.matmul(input.ConnectorWeight.transpose(0, 1))
            .add(input.ConnectorBias)
            .contiguous();
        var attentionMask = input.PromptAttentionMask.contiguous();

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["schema"] = Schema,
            ["sample_id"] = sampleId,
            ["num_frames"] = frames.ToString(CultureInfo.InvariantCulture),
            ["height"] = height.ToString(CultureInfo.InvariantCulture),
            ["width"] = width.ToString(CultureInfo.InvariantCulture),
            ["fps"] = input.FramesPerSecond.ToString("R", CultureInfo.InvariantCulture),
        };
        new SafeTensorFile(
            new Dictionary<string, SafeTensor>(StringComparer.Ordinal)
            {
                ["video_latents"] = SafeTensor.FromTorchTensor(encoded),
            },
            metadata).Save(videoPath);
        new SafeTensorFile(
            new Dictionary<string, SafeTensor>(StringComparer.Ordinal)
            {
                ["video_prompt_embeds"] = SafeTensor.FromTorchTensor(promptEmbeds),
                ["prompt_attention_mask"] = SafeTensor.FromTorchTensor(attentionMask),
            },
            metadata).Save(conditionsPath);
        return new TrainingPreprocessResult(videoPath, conditionsPath);
    }

    private static void Validate(TrainingPreprocessInput input)
    {
        if (input.Video.dim() != 4 || input.VideoEncoderWeight.dim() != 2 ||
            input.VideoEncoderBias.dim() != 1 || input.CaptionFeatures.dim() != 2 ||
            input.ConnectorWeight.dim() != 2 || input.ConnectorBias.dim() != 1 ||
            input.PromptAttentionMask.dim() != 1)
        {
            throw new ArgumentException("M6 preprocessing tensors have invalid ranks.", nameof(input));
        }
        if (input.Video.shape[0] != input.VideoEncoderWeight.shape[1] ||
            input.VideoEncoderWeight.shape[0] != input.VideoEncoderBias.shape[0])
        {
            throw new ArgumentException("Video encoder tensor dimensions are incompatible.", nameof(input));
        }
        if (input.CaptionFeatures.shape[1] != input.ConnectorWeight.shape[1] ||
            input.ConnectorWeight.shape[0] != input.ConnectorBias.shape[0] ||
            input.CaptionFeatures.shape[0] != input.PromptAttentionMask.shape[0] ||
            input.ConnectorWeight.shape[0] != input.VideoEncoderWeight.shape[0])
        {
            throw new ArgumentException("Caption connector tensor dimensions are incompatible.", nameof(input));
        }
        if (!double.IsFinite(input.FramesPerSecond) || input.FramesPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(input), "Frames per second must be positive and finite.");
        }
    }
}

public sealed class PrecomputedTrainingDataset
{
    private readonly string root;
    private readonly string[] sampleIds;

    public PrecomputedTrainingDataset(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        this.root = Path.GetFullPath(root);
        var videoDirectory = Path.Combine(this.root, "video_latents");
        var conditionsDirectory = Path.Combine(this.root, "conditions");
        if (!Directory.Exists(videoDirectory) || !Directory.Exists(conditionsDirectory))
        {
            throw new DirectoryNotFoundException("Precomputed data requires video_latents and conditions directories.");
        }

        var videos = Directory.EnumerateFiles(videoDirectory, "*.safetensors")
            .Select(path => Path.GetFileNameWithoutExtension(path)!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        var conditions = Directory.EnumerateFiles(conditionsDirectory, "*.safetensors")
            .Select(path => Path.GetFileNameWithoutExtension(path)!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        if (videos.Length == 0 || !videos.SequenceEqual(conditions, StringComparer.Ordinal))
        {
            throw new InvalidDataException("Precomputed video and condition sample sets must be non-empty and identical.");
        }
        sampleIds = videos;
    }

    public int Count => sampleIds.Length;

    public IReadOnlyList<string> SampleIds => sampleIds;

    public PrecomputedTrainingSample Load(int index, Device? device = null)
    {
        if ((uint)index >= (uint)sampleIds.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
        var sampleId = sampleIds[index];
        var videoFile = SafeTensorFile.Load(Path.Combine(root, "video_latents", sampleId + ".safetensors"));
        var conditionsFile = SafeTensorFile.Load(Path.Combine(root, "conditions", sampleId + ".safetensors"));
        ValidateMetadata(sampleId, videoFile.Metadata, conditionsFile.Metadata);
        if (!videoFile.Tensors.TryGetValue("video_latents", out var video) ||
            !conditionsFile.Tensors.TryGetValue("video_prompt_embeds", out var prompt) ||
            !conditionsFile.Tensors.TryGetValue("prompt_attention_mask", out var mask))
        {
            throw new InvalidDataException($"Precomputed sample '{sampleId}' is missing required tensors.");
        }
        return new PrecomputedTrainingSample(
            sampleId,
            video.ToTorchTensor(device),
            prompt.ToTorchTensor(device),
            mask.ToTorchTensor(device),
            int.Parse(videoFile.Metadata["num_frames"], CultureInfo.InvariantCulture),
            int.Parse(videoFile.Metadata["height"], CultureInfo.InvariantCulture),
            int.Parse(videoFile.Metadata["width"], CultureInfo.InvariantCulture),
            double.Parse(videoFile.Metadata["fps"], CultureInfo.InvariantCulture));
    }

    private static void ValidateMetadata(
        string sampleId,
        IReadOnlyDictionary<string, string> video,
        IReadOnlyDictionary<string, string> conditions)
    {
        var required = new[] { "schema", "sample_id", "num_frames", "height", "width", "fps" };
        foreach (var key in required)
        {
            if (!video.TryGetValue(key, out var left) || !conditions.TryGetValue(key, out var right) || left != right)
            {
                throw new InvalidDataException($"Precomputed sample '{sampleId}' metadata '{key}' is missing or mismatched.");
            }
        }
        if (video["schema"] != TrainingPreprocessor.Schema || video["sample_id"] != sampleId)
        {
            throw new InvalidDataException($"Precomputed sample '{sampleId}' schema metadata is invalid.");
        }
    }
}

public sealed class PrecomputedTrainingSample : IDisposable
{
    private bool disposed;

    public PrecomputedTrainingSample(
        string sampleId,
        Tensor videoLatents,
        Tensor videoPromptEmbeds,
        Tensor promptAttentionMask,
        int numFrames,
        int height,
        int width,
        double framesPerSecond)
    {
        SampleId = sampleId;
        VideoLatents = videoLatents;
        VideoPromptEmbeds = videoPromptEmbeds;
        PromptAttentionMask = promptAttentionMask;
        NumFrames = numFrames;
        Height = height;
        Width = width;
        FramesPerSecond = framesPerSecond;
    }

    public string SampleId { get; }
    public Tensor VideoLatents { get; }
    public Tensor VideoPromptEmbeds { get; }
    public Tensor PromptAttentionMask { get; }
    public int NumFrames { get; }
    public int Height { get; }
    public int Width { get; }
    public double FramesPerSecond { get; }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        VideoLatents.Dispose();
        VideoPromptEmbeds.Dispose();
        PromptAttentionMask.Dispose();
    }
}
