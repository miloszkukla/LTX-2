using System.Diagnostics;

namespace Ltx.Media;

internal static class ToolProcess
{
    public static byte[] CaptureBytes(string executable, IReadOnlyList<string> arguments)
    {
        using var output = new MemoryStream();
        Run(executable, arguments, input: null, output);
        return output.ToArray();
    }

    public static string CaptureText(string executable, IReadOnlyList<string> arguments)
    {
        var bytes = CaptureBytes(executable, arguments);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    public static void Run(
        string executable,
        IReadOnlyList<string> arguments,
        Stream? input = null,
        Stream? output = null)
    {
        var startInfo = new ProcessStartInfo(executable)
        {
            RedirectStandardInput = input is not null,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = startInfo };
        if (!process.Start())
        {
            throw new InvalidOperationException($"Could not start required media tool '{executable}'.");
        }

        var stderrTask = process.StandardError.ReadToEndAsync();
        var outputTask = output is null
            ? process.StandardOutput.BaseStream.CopyToAsync(Stream.Null)
            : process.StandardOutput.BaseStream.CopyToAsync(output);
        Task inputTask = Task.CompletedTask;
        if (input is not null)
        {
            inputTask = CopyInputAsync(input, process.StandardInput.BaseStream);
        }

        Task.WhenAll(stderrTask, outputTask, inputTask).GetAwaiter().GetResult();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            var stderr = stderrTask.Result.Trim();
            if (stderr.Length > 800)
            {
                stderr = stderr[..800];
            }
            throw new InvalidDataException($"Media tool '{executable}' exited {process.ExitCode}: {stderr}");
        }
    }

    private static async Task CopyInputAsync(Stream source, Stream destination)
    {
        await source.CopyToAsync(destination).ConfigureAwait(false);
        await destination.FlushAsync().ConfigureAwait(false);
        destination.Close();
    }
}
