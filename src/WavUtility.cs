using System;
using System.IO;
using UnityEngine;

public static class WavUtility
{
    const int HEADER_SIZE = 44;

    // Metodo per convertire un AudioClip in byte[]
    public static byte[] FromAudioClip(AudioClip clip)
    {
        using (MemoryStream stream = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            WriteWavHeader(writer, clip);
            WriteWavData(writer, clip);

            // Torna indietro e aggiorna la dimensione dei dati
            long fileSize = stream.Position;
            stream.Seek(4, SeekOrigin.Begin);
            writer.Write((int)(fileSize - 8)); // Dimensione file
            stream.Seek(40, SeekOrigin.Begin);
            writer.Write((int)(fileSize - HEADER_SIZE)); // Dimensione dati

            return stream.ToArray();
        }
    }

    // Metodo per scrivere l'header WAV
    private static void WriteWavHeader(BinaryWriter writer, AudioClip clip)
    {
        int sampleRate = clip.frequency;
        int channels = clip.channels;
        int bitsPerSample = 16; // Standard per WAV
        int byteRate = sampleRate * channels * bitsPerSample / 8;

        writer.Write("RIFF".ToCharArray());
        writer.Write(0); // Placeholder per la dimensione del file
        writer.Write("WAVE".ToCharArray());
        writer.Write("fmt ".ToCharArray());
        writer.Write(16); // Dimensione del chunk fmt
        writer.Write((short)1); // Audio PCM
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(byteRate);
        writer.Write((short)(channels * bitsPerSample / 8)); // Block Align
        writer.Write((short)bitsPerSample);
        writer.Write("data".ToCharArray());
        writer.Write(0); // Placeholder per la dimensione dei dati
    }

    // Metodo per scrivere i dati audio
    private static void WriteWavData(BinaryWriter writer, AudioClip clip)
    {
        float[] samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);

        foreach (float sample in samples)
        {
            // Converte da float (-1.0f a 1.0f) a short (-32768 a 32767)
            short intSample = (short)(sample * short.MaxValue);
            writer.Write(intSample);
        }
    }
}
