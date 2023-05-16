import soundfile as sf
from scipy.signal import resample
from utilities import inputwav
import numpy as np


def slow(filename, p=10, wout=True):
    """
    Resamples the input by p%, slowing it down by p%. Uses the scipy resample
    function.

    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    p: float/int, optional, default=10
        percentage the audio is slowed by.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    f: resampled data in bits
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    if p > 0:
        print("Slowing...")
    if p < 0:
        print(
            "Warning: You are speeding up the audio! Use positive value"
            + " for p to slow."
        )
    f = resample(data, int(len(data) * (1 + p / 100.0)))
    if wout:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_slow.wav", f, sr, "PCM_16")
        print("Done!")
    return f


def stereo(filename, time, wout=True):
    """
    Produces stereo effect. If file is mono, two channels are created and the
    R channel is delayed to simulate stereo width. Beware of phase issues.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    time : scalar (ms)
        Amount of time the right channel is delayed by, in milliseconds.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.

    Returns
    -------
    data_st: array containing the stereo waveform in normalized bits.
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    s_shift = int(sr * time * 1e-3)
    R = np.zeros(n)
    L = np.zeros(n)
    if ch == 2:
        L[:] = data[:, 0]
        R[:] = data[:, 1]
    if ch == 1:
        L[:] = data[:, 0]
        R[:] = data[:, 0]
    print("Applying stereo width...")
    for i in range(n - s_shift):
        R[i] = R[i + s_shift]
    data_st = np.zeros((n, 2))
    data_st[:, 0] = L[:]
    data_st[:, 1] = R[:]
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_stereo.wav", data_st, 44100, "PCM_16"
        )
    print("Done!")
    return data_st
