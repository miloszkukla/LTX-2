using System.Globalization;
using System.Text.Json;
using Ltx.Media;
using Ltx.Pipelines;

namespace Ltx.Cli;

public static class PipelineCli
{
    public static int Run(IReadOnlyList<string> arguments, TextWriter standardOutput, TextWriter standardError)
    {
        try
        {
            if (arguments.Count == 1 && arguments[0] is "-h" or "--help")
            {
                WriteRootHelp(standardOutput);
                return 0;
            }
            if (arguments.Count == 1 && arguments[0] == "--version")
            {
                standardOutput.WriteLine("ltx-csharp M5");
                return 0;
            }
            if (arguments.Count == 0)
            {
                throw new CliUsageException("a pipeline mode is required");
            }

            PipelineMode mode;
            try
            {
                mode = PipelineMode.Resolve(arguments[0]);
            }
            catch (ArgumentException)
            {
                throw new CliUsageException($"unknown pipeline mode '{arguments[0]}'");
            }

            if (arguments.Skip(1).Any(value => value is "-h" or "--help"))
            {
                WriteModeHelp(mode, standardOutput);
                return 0;
            }

            var options = OptionSet.Parse(arguments.Skip(1).ToArray());
            var request = BuildRequest(mode, options);
            var result = PipelineRegistry.Create(mode).Execute(request);
            standardOutput.WriteLine(JsonSerializer.Serialize(new
            {
                mode = result.Mode,
                outputs = result.OutputPaths,
                frames = result.FrameCount,
                fps = result.FramesPerSecond,
                has_audio = result.HasAudio,
                hdr = result.IsHdr,
                execution = result.Execution,
            }));
            return 0;
        }
        catch (CliUsageException exception)
        {
            standardError.WriteLine($"error: {exception.Message}");
            standardError.WriteLine("Use --help for usage.");
            return 2;
        }
        catch (Exception exception) when (exception is ArgumentException or FileNotFoundException or InvalidDataException)
        {
            standardError.WriteLine($"error: {exception.Message}");
            return 1;
        }
    }

    private static PipelineRequest BuildRequest(PipelineMode mode, OptionSet options)
    {
        if (!options.Has("--fixture-mode"))
        {
            RequireModelLayout(mode, options);
        }

        var frameRate = options.Double("--frame-rate", 8);
        var hdr = ParseHdr(options.Optional("--hdr"));
        var videoPath = options.Optional("--video-path");
        if (videoPath is not null && Directory.Exists(videoPath))
        {
            if (!Directory.EnumerateFiles(videoPath, "*.exr").Any())
            {
                throw new CliUsageException("--video-path directory contains no EXR frames");
            }
            if (!options.Has("--frame-rate"))
            {
                throw new CliUsageException("--frame-rate is required for an EXR sequence");
            }
            if (hdr is null)
            {
                throw new CliUsageException("EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT}");
            }
        }
        else if (videoPath is not null && options.Has("--frame-rate") && mode.ModuleName == "ltx_pipelines.retake")
        {
            throw new CliUsageException("--frame-rate is only valid when --video-path is an EXR-frame folder");
        }

        var conditioning = options.Optional("--video-conditioning");
        if (conditioning is not null && Directory.Exists(conditioning) && hdr is null)
        {
            throw new CliUsageException("EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT}");
        }

        var request = new PipelineRequest
        {
            Prompt = options.Optional("--prompt") ?? "",
            Seed = options.Integer("--seed", 20260820),
            Width = options.Integer("--width", 64),
            Height = options.Integer("--height", 64),
            FrameCount = options.Integer("--num-frames", 9),
            FramesPerSecond = frameRate,
            OutputPath = options.Optional("--output-path"),
            OutputDirectory = options.Optional("--output-dir"),
            AudioPath = options.Optional("--audio-path"),
            ReferenceVideoPath = options.Optional("--reference-video"),
            VideoPath = videoPath,
            ConditioningPath = conditioning,
            InputPath = options.Optional("--input"),
            StartTime = options.Double("--start-time", 0),
            EndTime = options.Double("--end-time", 0),
            HdrColorSpace = hdr,
            SkipPreview = options.Has("--skip-mp4"),
            FixtureMode = options.Has("--fixture-mode"),
            CheckpointPath = options.Optional("--distilled-checkpoint-path") ??
                options.Optional("--checkpoint-path") ?? options.Optional("--transformer-path"),
            TextEmbeddingsPath = options.Optional("--text-embeddings"),
            TorchSharpLibraryPath = Environment.GetEnvironmentVariable("LTX_TORCHSHARP_LIBRARY"),
            InferenceSteps = options.Integer("--num-inference-steps", 8),
        };
        if (!request.FixtureMode && request.TextEmbeddingsPath is null)
        {
            throw new CliUsageException("missing --text-embeddings for native checkpoint inference");
        }
        ValidateRequired(mode, request);
        return request;
    }

    private static void RequireModelLayout(PipelineMode mode, OptionSet options)
    {
        var distilled = mode.ModuleName is "ltx_pipelines.distilled" or "ltx_pipelines.dubit" or
            "ltx_pipelines.hdr_ic_lora" or "ltx_pipelines.ic_lora" or "ltx_pipelines.retake";
        var checkpointFlag = distilled ? "--distilled-checkpoint-path" : "--checkpoint-path";
        if (!options.Has(checkpointFlag) && !options.Has("--transformer-path"))
        {
            throw new CliUsageException($"missing {checkpointFlag} (monolith) or --transformer-path (split)");
        }
    }

    private static void ValidateRequired(PipelineMode mode, PipelineRequest request)
    {
        if (mode.HdrBatch)
        {
            Require(request.InputPath, "--input");
            Require(request.OutputDirectory, "--output-dir");
            return;
        }
        Require(request.OutputPath, "--output-path");
        if (mode.ModuleName == "ltx_pipelines.a2vid_two_stage")
        {
            Require(request.AudioPath, "--audio-path");
        }
        else if (mode.ModuleName == "ltx_pipelines.dubit")
        {
            Require(request.ReferenceVideoPath, "--reference-video");
        }
        else if (mode.ModuleName == "ltx_pipelines.ic_lora")
        {
            Require(request.ConditioningPath, "--video-conditioning");
        }
        else if (mode.ModuleName == "ltx_pipelines.retake")
        {
            Require(request.VideoPath, "--video-path");
            if (request.StartTime >= request.EndTime)
            {
                throw new CliUsageException("--start-time must be less than --end-time");
            }
        }
    }

    private static HdrColorSpace? ParseHdr(string? value) => value?.ToUpperInvariant() switch
    {
        null => null,
        "SRGB_LINEAR" => HdrColorSpace.SrgbLinear,
        "ACESCG" => HdrColorSpace.AcesCg,
        "ACESCCT" => HdrColorSpace.AcesCct,
        _ => throw new CliUsageException("invalid --hdr value; choose SRGB_LINEAR, ACESCG, or ACESCCT"),
    };

    private static void Require(string? value, string flag)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new CliUsageException($"missing required argument {flag}");
        }
    }

    private static void WriteRootHelp(TextWriter writer)
    {
        writer.WriteLine("Usage: ltx <pipeline-mode> [options]");
        writer.WriteLine("In-scope single-GPU pipeline modes:");
        foreach (var mode in PipelineMode.InScope)
        {
            writer.WriteLine($"  {mode.CommandName,-26} {mode.PipelineName}");
        }
    }

    private static void WriteModeHelp(PipelineMode mode, TextWriter writer)
    {
        writer.WriteLine($"Usage: ltx {mode.CommandName} [options]");
        writer.WriteLine($"Python-compatible mode: {mode.ModuleName}");
        writer.WriteLine("Common options: --prompt --seed --output-path --height --width --num-frames --frame-rate");
        writer.WriteLine("Media options: --image --video-conditioning --video-path --audio-path --hdr");
        writer.WriteLine("Model options: --checkpoint-path/--distilled-checkpoint-path or split model paths");
    }

    private sealed class CliUsageException(string message) : Exception(message);

    private sealed class OptionSet(Dictionary<string, List<string[]>> values)
    {
        private static readonly HashSet<string> BooleanFlags =
        [
            "--fixture-mode", "--skip-mp4", "--skip-stage-2", "--high-quality", "--enhance-prompt",
            "--enhance-static-cache", "--include-audio", "--compile-transformer", "--compile-video-vae",
        ];

        private static readonly Dictionary<string, (int Minimum, int Maximum)> MultiValueFlags = new()
        {
            ["--image"] = (3, 4),
            ["--video-conditioning"] = (2, 2),
            ["--conditioning-attention-mask"] = (2, 2),
            ["--lora"] = (1, 2),
            ["--distilled-lora"] = (1, 2),
            ["--detailing-lora"] = (1, 2),
            ["--auto-duration"] = (2, 2),
            ["--video-stg-blocks"] = (0, int.MaxValue),
            ["--audio-stg-blocks"] = (0, int.MaxValue),
            ["--compile"] = (0, int.MaxValue),
        };

        private static readonly HashSet<string> ScalarFlags =
        [
            "--prompt", "--negative-prompt", "--seed", "--height", "--width", "--num-frames", "--frame-rate",
            "--output-path", "--output-dir", "--audio-path", "--audio-start-time", "--audio-max-duration",
            "--reference-video", "--reference-strength", "--video-path", "--start-time", "--end-time",
            "--input", "--hdr", "--hdr-lora", "--text-embeddings", "--spatial-upsampler-path",
            "--temporal-upsampler-path", "--temporal-upsample-rounds", "--distilled-checkpoint-path",
            "--checkpoint-path", "--transformer-path", "--text-encoder-path", "--video-vae-path", "--audio-vae-path",
            "--vocoder-path", "--gemma-root", "--prompt-enhancer-gemma-root", "--duration-head-path",
            "--offload-mode", "--quantization", "--diffvae-optimization", "--max-batch-size", "--num-inference-steps",
            "--num-generated-keyframes", "--video-cfg-guidance-scale", "--video-stg-guidance-scale",
            "--video-rescale-scale", "--a2v-guidance-scale", "--video-skip-step", "--audio-cfg-guidance-scale",
            "--audio-stg-guidance-scale", "--audio-rescale-scale", "--v2a-guidance-scale", "--audio-skip-step",
            "--conditioning-scale", "--spatial-tile", "--distilled-lora-strength-stage-1",
            "--distilled-lora-strength-stage-2",
        ];

        public static OptionSet Parse(IReadOnlyList<string> arguments)
        {
            var parsed = new Dictionary<string, List<string[]>>(StringComparer.Ordinal);
            for (var index = 0; index < arguments.Count;)
            {
                var flag = arguments[index];
                if (!flag.StartsWith("--", StringComparison.Ordinal))
                {
                    throw new CliUsageException($"unexpected positional argument '{flag}'");
                }
                index++;
                string[] items;
                if (BooleanFlags.Contains(flag))
                {
                    items = [];
                }
                else if (MultiValueFlags.TryGetValue(flag, out var arity))
                {
                    var start = index;
                    while (index < arguments.Count && !arguments[index].StartsWith("--", StringComparison.Ordinal) &&
                           index - start < arity.Maximum)
                    {
                        index++;
                    }
                    items = arguments.Skip(start).Take(index - start).ToArray();
                    if (items.Length < arity.Minimum)
                    {
                        throw new CliUsageException($"{flag} requires at least {arity.Minimum} value(s)");
                    }
                }
                else if (ScalarFlags.Contains(flag))
                {
                    if (index >= arguments.Count || arguments[index].StartsWith("--", StringComparison.Ordinal))
                    {
                        throw new CliUsageException($"{flag} requires one value");
                    }
                    items = [arguments[index++]];
                }
                else
                {
                    throw new CliUsageException($"unrecognized argument {flag}");
                }
                if (!parsed.TryGetValue(flag, out var occurrences))
                {
                    occurrences = [];
                    parsed.Add(flag, occurrences);
                }
                occurrences.Add(items);
            }
            return new OptionSet(parsed);
        }

        public bool Has(string flag) => values.ContainsKey(flag);

        public string? Optional(string flag) => values.TryGetValue(flag, out var occurrences) && occurrences[^1].Length > 0
            ? occurrences[^1][0]
            : null;

        public int Integer(string flag, int defaultValue)
        {
            var value = Optional(flag);
            return value is null ? defaultValue : int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result)
                ? result
                : throw new CliUsageException($"{flag} expects an integer");
        }

        public double Double(string flag, double defaultValue)
        {
            var value = Optional(flag);
            return value is null ? defaultValue : double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result)
                ? result
                : throw new CliUsageException($"{flag} expects a number");
        }
    }
}
