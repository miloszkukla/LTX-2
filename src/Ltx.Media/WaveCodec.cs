using System.Buffers.Binary;
using System.Text;

namespace Ltx.Media;

public static class WaveCodec
{
    public static void WritePcm16(string path, AudioData audio)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.SampleRate <= 0 || audio.Channels <= 0 || audio.Samples.Length % audio.Channels != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(audio), "Audio layout must have a positive rate and complete frames.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);
        var dataBytes = checked(audio.Samples.Length * sizeof(short));
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataBytes);
        writer.Write(Encoding.ASCII.GetBytes("WAVEfmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)audio.Channels);
        writer.Write(audio.SampleRate);
        writer.Write(audio.SampleRate * audio.Channels * sizeof(short));
        writer.Write((short)(audio.Channels * sizeof(short)));
        writer.Write((short)16);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataBytes);
        foreach (var sample in audio.Samples)
        {
            writer.Write((short)Math.Clamp(MathF.Round(sample * 32767f), short.MinValue, short.MaxValue));
        }
    }

    public static AudioData ReadPcm16(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 44 || Encoding.ASCII.GetString(bytes, 0, 4) != "RIFF" ||
            Encoding.ASCII.GetString(bytes, 8, 4) != "WAVE")
        {
            throw new InvalidDataException("Input is not a RIFF/WAVE file.");
        }

        var offset = 12;
        short format = 0;
        short channels = 0;
        var sampleRate = 0;
        short bits = 0;
        ReadOnlySpan<byte> pcm = default;
        while (offset + 8 <= bytes.Length)
        {
            var id = Encoding.ASCII.GetString(bytes, offset, 4);
            var size = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            offset += 8;
            if (size < 0 || offset + size > bytes.Length)
            {
                throw new InvalidDataException("WAVE chunk extends past the end of the file.");
            }
            if (id == "fmt ")
            {
                format = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(offset, 2));
                channels = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(offset + 2, 2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
                bits = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(offset + 14, 2));
            }
            else if (id == "data")
            {
                pcm = bytes.AsSpan(offset, size);
            }
            offset += size + (size & 1);
        }

        if (format != 1 || bits != 16 || channels <= 0 || sampleRate <= 0 || pcm.IsEmpty)
        {
            throw new InvalidDataException("Only non-empty 16-bit PCM WAVE input is supported.");
        }

        var samples = new float[pcm.Length / sizeof(short)];
        for (var index = 0; index < samples.Length; index++)
        {
            samples[index] = BinaryPrimitives.ReadInt16LittleEndian(pcm.Slice(index * 2, 2)) / 32768f;
        }
        return new AudioData(samples, sampleRate, channels);
    }
}
