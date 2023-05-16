import numpy as np
import pylab as py
from scipy.fftpack import fft, ifft, rfft, irfft
from scipy.signal import (
    butter,
    lfilter,
)
from scipy.signal import freqz
import soundfile as sf
import time

from utilities import inputwav


def FFT_brickwallHPF(filename, cutoff, wout=True, plot=True):
    """
    Deletes frequencies below cutoff using brickwall. Beware of phase issues.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    cutoff: int (Hz)
        Frequency of the filter cutoff below which data is filtered out.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    data_filtered: array containing filtered audio in bits
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    print("Applying FFT...")
    yfreq = rfft(data, axis=0)
    xfreq = np.linspace(0, sr / (2.0), n)
    yfreqBHPF = np.zeros((n, ch))
    yfreqBHPF[0:n, :] = yfreq
    print("Applying brickwall at " + str(cutoff) + " Hz...")
    yfreqBHPF[0 : np.searchsorted(xfreq, cutoff), :] = 0.0
    data_filtered = irfft(yfreqBHPF, axis=0)
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_brickwallHPF.wav",
            data_filtered,
            sr,
            "PCM_16",
        )
    if plot:
        print("Plotting...")
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(xfreq, 20 * np.log10(abs(yfreq[0:n, 0] + 0.0001)), "k-", lw=0.5)
        ax1.semilogx(
            xfreq, 20 * np.log10(abs(yfreqBHPF[0 : n // 1, 0] + 0.0001)), "m-", lw=0.1
        )
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax2.plot(data, "k-", label="Raw")
        ax2.plot(data_filtered, "m-", label="Filtered")
        ax2.set_xlim(0, 10000)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2)
    print("Done!")
    return data_filtered


def FFT_brickwallBR(filename, start, stop, wout=True, plot=True):
    """
    Deletes frequencies below cutoff using brickwall. Beware of phase issues.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    cutoff: int (Hz)
        Frequency of the filter cutoff below which data is filtered out.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    data_filtered: array containing filtered audio in bits
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    print("Applying FFT...")
    yfreq = fft(data, axis=0)
    xfreq = np.linspace(0, sr / (2.0), n // 2)
    yfreqBHPF = np.zeros((n, ch), dtype=complex)
    yfreqBHPF[0:n, :] = yfreq
    print("Applying brickwall centered at " + str((start + stop) / 2) + " Hz...")
    yfreqBHPF[np.searchsorted(xfreq, start) : np.searchsorted(xfreq, stop), :] = 0.00001
    data_filtered = ifft(yfreqBHPF, axis=0)
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_brickwallHPF.wav",
            data_filtered.real,
            sr,
            "PCM_16",
        )
    if plot:
        print("Plotting...")
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(xfreq, 20 * np.log10(abs(yfreq[0 : n // 2])), "k-", lw=0.5)
        ax1.semilogx(xfreq, 20 * np.log10(abs(yfreqBHPF[0 : n // 2, 0])), "m-", lw=0.1)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude (dB)")
        ax2.plot(data, "k-")
        ax2.plot(data_filtered, "m-")
        ax2.set_xlim(0, 1000)
        # ax2.set_ylim(-1,1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2)
    print("Done!")
    return data_filtered


def FFT_brickwallLPF(filename, cutoff, wout=True, plot=True):
    """
    Deletes frequencies above cutoff using brickwall.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples

    data: array containing the signal in bits.

    data_dB: array containing the signal in dB.

    sr: sample rate

    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    print("Applying FFT...")
    W = np.zeros((n, 2))
    W[:, 0] = 1  # blackman(n)
    W[:, 1] = 1  # blackman(n)
    yfreq = rfft(data * W, axis=0)
    xfreq = np.linspace(0, sr / (2.0), n // 1)
    yfreqBLPF = np.zeros((n, ch))
    yfreqBLPF[0:n, :] = yfreq
    print("Applying brickwall at " + str(cutoff) + " Hz...")
    yfreqBLPF[n : np.searchsorted(xfreq, cutoff) : -1, :] = 0.0
    data_filtered = irfft(yfreqBLPF, axis=0)
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_brickwallLPF.wav",
            data_filtered,
            sr,
            "PCM_16",
        )
    if plot:
        print("Plotting...")
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(
            xfreq, 20 * np.log10(abs(yfreq[0 : n // 1, :] + 0.0001)), "k-", lw=0.5
        )
        ax1.semilogx(
            xfreq, 20 * np.log10(abs(yfreqBLPF[0 : n // 1, :] + 0.0001)), "m-", lw=0.1
        )
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude (dB)")
        ax2.plot(data, "k-", label="Raw")
        ax2.plot(data_filtered, "m-", lw=1, label="Filtered")
        ax2.set_xlim(0, 10000)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2, frameon=False, ncol=2)
    print("Done!")
    end = time.time()
    elapsed = end - start
    print("Completed in " + str(elapsed) + " seconds.")
    return data_filtered


def LPF(filename, cutoff, Q=1, wout=True, plot=True):
    """
    Lowpass filter.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.

    Q:  int, optional, default=1
        Number of poles in filter or steepness of filter edge.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples

    data: array containing the signal in bits.

    data_dB: array containing the signal in dB.

    sr: sample rate

    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    b, a = butter(Q, cutoff / sr, btype="low")
    data_filtered = lfilter(b, a, data, axis=0)
    print("Applying FFT...")
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_LPF.wav", data_filtered, sr, "PCM_16"
        )
    if plot:
        print("Plotting...")
        py.close()
        w, h = freqz(b, a, worN=1024)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5 * sr * w / np.pi, abs(h), "k--")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Rel. Amplitude")
        ax1.grid()
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(1, 20000)
        ax2.plot(data, "k-", label="Raw data")
        ax2.plot(data_filtered, "m-", lw=1, label="Filtered data")
        ax2.set_xlim(0, 10000)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2, frameon=False, ncol=2)
        py.subplots_adjust(hspace=0.35)
    print("Done!")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return data_filtered


def HPF(filename, cutoff, Q=1, wout=True, plot=True):
    """
    Deletes frequencies above cutoff using brickwall.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples

    data: array containing the signal in bits.

    data_dB: array containing the signal in dB.

    sr: sample rate

    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    b, a = butter(Q, cutoff / sr, btype="high")
    data_filtered = lfilter(b, a, data, axis=0)
    print("Applying FFT...")
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_HPF.wav", data_filtered, sr, "PCM_16"
        )
    if plot:
        print("Plotting...")
        py.close()
        w, h = freqz(b, a, worN=16384)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5 * sr * w / np.pi, abs(h), "k--")
        ax1.set_title("Fiter Frequency Response")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Rel. Amplitude")
        ax1.grid()
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(1, 20000)
        ax2.plot(data, "k-", label="Raw data")
        ax2.plot(data_filtered, "m-", lw=1, label="Filtered data")
        ax2.set_xlim(0, 10000)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2, frameon=False, ncol=2)
        py.subplots_adjust(hspace=0.35)
    print("Done!")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return data_filtered


def bandpass(filename, f1, f2, Q, wout=True, plot=True):
    """
    Filters frequency content outside of [f1,f2] domain.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    f1: int (Hz)
        Start frequency of the filter cutoff above which data is filtered out.

    f2: int (Hz)
        Stop frequency of the filter cutoff below which data is filtered out.

    Q:  int, optional, default=1
        Number of poles in filter or steepness of filter edge.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.

    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples

    data: array containing the signal in bits.

    data_dB: array containing the signal in dB.

    sr: sample rate

    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    b, a = butter(Q, Wn=(f1 / sr, f2 / sr), btype="bandpass")
    data_filtered = lfilter(b, a, data, axis=0)
    print("Applying FFT...")
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_BP.wav", data_filtered, sr, "PCM_16"
        )
    if plot:
        print("Plotting...")
        py.close()
        w, h = freqz(b, a, worN=16384)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5 * sr * w / np.pi, abs(h), "k-")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Rel. Amplitude")
        ax1.grid()
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(1, 20000)
        ax2.plot(data, "k-", label="Raw data")
        ax2.plot(data_filtered, "m-", lw=1, label="Filtered data")
        ax2.set_xlim(0, 10000)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("Amplitude (Norm Bits)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc=2, frameon=False, ncol=2)
        py.subplots_adjust(hspace=0.35)
    print("Done!")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return data_filtered
