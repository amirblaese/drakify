"""
Created on Sat Jun 06 20:47:37 2020
@author: Shayan
"""
import numpy as np
import pylab as py
from scipy.fftpack import fft, ifft, rfft, irfft
from scipy.signal import (
    blackman,
    hamming,
    chebwin,
    resample,
    stft,
    butter,
    lfilter,
    convolve,
)
from scipy.signal import freqz
import soundfile as sf
import os
import time


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
    if wout == True:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_slow.wav", f, sr, "PCM_16")
        print("Done!")
    return f


def verb_delay(
    filename, l, t, d, wout=True
):  # l = predelay   d= decay smaller = less decay, t= number of delays
    # low l turns into chorus
    """
    Repeats sound to form delay or reverb with appropriate parameters.

    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    l:  int
        Spacing between samples in units of sample. Serves as "predelay"
        or the delay spacing.

    t:  int
        Number of delays or repetitions produced.

    d:  float
        Characteristic decay time of the feedback.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    data_verb: Data with reverb added in bit units.
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    data_ex = np.zeros(((n + l * t), ch))
    data_ex[0:n, :] = data
    data_Rex = np.zeros((len(data_ex), t, ch))
    print("Applying reverb...")
    for k in range(ch):
        for i in range(len(data)):
            for j in range(t):
                data_Rex[i + l * (j + 1), j, k] = data_ex[i, k] * np.exp(-d * (j + 1))
    data_F = data_ex
    print("Mixing...")
    for i in range(t):
        data_F = data_F + 1 * data_Rex[:, i, :]
    data_F = 1 * data_F
    data_verb = data_F + data_ex
    if wout == True:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_verbed.wav", data_verb, sr, "PCM_16"
        )
    print("Done!")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return data_verb


def distort(filename, threshold=0.25, type="arctan", wout=True, plot=False):
    """
    Applies distortion to signal. Reshapes the signal above user input
    threshold (in bits).
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    threshold : scalar
        lower = more distortion, try 0.25 for heavy distortion.

    type: string, optional, default=cubic
        Type of distortion that is applied. Default is arctan. more will be added
          soon.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.

    plot: True/False, optional, default =True
        Produces plot of input and output waveforms.

    Returns
    -------
    dataD: array containing the limited waveform in bits
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    dataD = np.zeros((len(data), ch))
    dataD[: len(data), :] = data  # data_dB
    if type == "arctan":
        print("Applying arctan distortion...")
        for k in range(ch):
            for i in range(0, len(data)):
                dataD[i, k] = (2 / np.pi) * np.arctan((np.pi / threshold) * dataD[i, k])
    if wout == True:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_distort.wav", dataD, sr, "PCM_16")
    # PLOTTING#
    if plot == True:
        print("Plotting...")
        t = np.linspace(0, n / (1.0 * sr), n)
        py.close()
        fig, (ax1) = py.subplots(nrows=1)
        ax1.plot(t, data, "k-", linewidth=1, label=filename)
        ax1.plot(t, dataD, "m-", linewidth=0.5, label=filename + " distorted")
        ax1.axhline(threshold, linestyle="-", color="cyan", linewidth=1)
        ax1.axhline(-threshold, linestyle="-", color="cyan", linewidth=1)
        ax1.legend(loc=1)
        ax1.set_ylabel("Amplitude (Rel. Bit)")
        ax1.set_xlabel("Time (s)")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("Done!")
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return dataD


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
    if wout == True:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_stereo.wav", data_st, 44100, "PCM_16"
        )
    print("Done!")
    return data_st


def mono(filename, wout=True):
    """
    Converts a stereo track to mono.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.

    Returns
    -------
    data_m: array containing the stereo waveform in normalized bits.
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    if ch == 2:
        print("Converting to mono...")
        L = data[:, 0]
        R = data[:, 1]
        n = len(data)
        data_m = np.zeros((n, 1))
        data_m = L / 2.0 + R / 2.0
        if wout == True:
            print("Exporting...")
            sf.write(
                filename[0 : len(filename) - 4] + "_mono.wav", data_m, sr, "PCM_16"
            )
        print("Done!")
        return data_m
    else:
        print("Error: input is already mono stoooooooooopid!")


def drakify(filename, wet=0.35):
    # compress(filename,1,1,-20,1,1,wout=True,plot=False)
    slow(filename[0 : len(filename) - 4] + ".wav", wout=True, p=20)
    conv_reverb(filename[0 : len(filename) - 4] + "_slow.wav", wet=wet)
    os.replace(
        filename[0 : len(filename) - 4] + "_slow_verbed.wav",
        filename[0 : len(filename) - 4] + "_drakify.wav",
    )
    os.remove(filename[0 : len(filename) - 4] + "_slow.wav")
    # os.remove(filename[0:len(filename)-4]+'_compressed.wav')
    normalize(filename[0 : len(filename) - 4] + "_drakify.wav")
    os.remove(filename[0 : len(filename) - 4] + "_drakify.wav")
    # print(3)


def drakifyD(filename):
    slow(filename, wout=True)
    distort(filename[0 : len(filename) - 4] + "_slow.wav", 0.1, type="flat")
    conv_reverb(filename[0 : len(filename) - 4] + "_slow_distort.wav")
    os.replace(
        filename[0 : len(filename) - 4] + "_slow_distort_verbed.wav",
        filename[0 : len(filename) - 4] + "_drakifyD.wav",
    )
    os.remove(filename[0 : len(filename) - 4] + "_slow.wav")
    # print(3)


def drakifyL(filename):
    slow(filename, wout=True)
    limit(filename[0 : len(filename) - 4] + "_slow.wav", -20, 7)
    conv_reverb(filename[0 : len(filename) - 4] + "_slow_limit.wav")
    os.replace(
        filename[0 : len(filename) - 4] + "_slow_limit_verbed.wav",
        filename[0 : len(filename) - 4] + "_drakifyL.wav",
    )
    os.remove(filename[0 : len(filename) - 4] + "_slow.wav")
    # print(3)


def robot(filename, wout=True):
    verb_delay(filename, 1000, 10, 0.000000009)
    os.replace(
        filename[0 : len(filename) - 4] + "_verbed.wav",
        filename[0 : len(filename) - 4] + "_robot.wav",
    )


def choverb(filename, wout=True):
    verb_delay(filename, 100, 1, 0.00000000001)
    os.replace(
        filename[0 : len(filename) - 4] + "_verbed.wav",
        filename[0 : len(filename) - 4] + "_choverb.wav",
    )


def verberator(filename, wout=True):
    verb_delay(filename, 10000, 3, 0.01)
    os.replace(
        filename[0 : len(filename) - 4] + "_verbed.wav",
        filename[0 : len(filename) - 4] + "_verberator.wav",
    )


def robot2(filename, wout=True):
    verb_delay(filename, 500, 10, 0.000001)
    os.replace(
        filename[0 : len(filename) - 4] + "_verbed.wav",
        filename[0 : len(filename) - 4] + "_robot2.wav",
    )


def long(filename, wout=True):
    verb_delay(filename, 5000, 20, 0.1)
    os.replace(
        filename[0 : len(filename) - 4] + "_verbed.wav",
        filename[0 : len(filename) - 4] + "_long.wav",
    )


def telephone(filename, wout=True):
    bandpass(filename, 300, 3400, 3)
    os.replace(
        filename[0 : len(filename) - 4] + "_BP.wav",
        filename[0 : len(filename) - 4] + "_telephone.wav",
    )


def bluetoothspeaker(filename):
    normalize(filename)
    LPF(filename[0 : len(filename) - 4] + "_normalized.wav", 400, Q=2)
    distort(
        filename[0 : len(filename) - 4] + "_normalized_LPF.wav", 0.41, type="digital"
    )
    os.replace(
        filename[0 : len(filename) - 4] + "_normalized_LPF_distort.wav",
        filename[0 : len(filename) - 4] + "_btspeaker.wav",
    )
    os.remove(filename[0 : len(filename) - 4] + "_normalized_LPF.wav")
    os.remove(filename[0 : len(filename) - 4] + "_normalized.wav")
