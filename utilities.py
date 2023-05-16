import soundfile as sf
import numpy as np


def inputwav(filename):
    """
    Decodes input file using "soundfile" library. Determines if file is
    mono or stereo and converts data to dB.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
    Returns
    -------
    n: length of audio in samples

    data: array containing the signal in bits.

    data_dB: array containing the signal in dB.

    sr: sample rate

    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    data, sr = sf.read(filename)
    try:
        ch = len(
            data[
                0,
            ]
        )
    except Exception:
        ch = 1
    # Reshape the data so other functions can interpret the array if mono.
    # basically transposing the data
    if ch == 1:
        data = data.reshape(-1, 1)
    n = len(data)
    # This prevents log(data) producing nan when data is 0
    data[np.where(data == 0)] = 0.00001
    # convert to dB
    data_dB = 20 * np.log10(abs(data))
    return n, data, data_dB, sr, ch


def normalize(filename, wout=True):
    """
    Normalize volume to 0 dB.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    data_norm: Normalized data array in bits
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    if ch == 1:
        diff = 0 - max(data_dB)
    if ch == 2:
        d1 = 0 - max(data_dB[:, 0])
        d2 = 0 - max(data_dB[:, 1])
        diff = max(d1, d2)
    data_dB_norm = data_dB + diff
    data_norm = 10.0 ** ((data_dB_norm) / 20.0)
    # sign the bits appropriately:
    for k in range(ch):
        for i in range(n):
            if data[i, k] < 0.0:
                data_norm[i, k] = -1.0 * data_norm[i, k]
    if wout:
        sf.write(
            filename[0 : len(filename) - 4] + "_normalized.wav", data_norm, sr, "PCM_16"
        )
    return data_norm


def view(filename):
    """
    Plots waveform of input audio file. Note that every 100th data point is
    plotted for faster performance
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    """
    n, data, data_dB, sr, ch = inputwav(filename)
    t = np.linspace(0, n / sr, n)
    py.close()
    fig, (ax1) = py.subplots(nrows=1)
    ax1.plot(t[0:n:100], data[0:n:100], "k-", linewidth=1, label=filename)
    ax1.legend(loc=1)
    ax1.set_ylabel("Amplitude (Rel. Bit)")
    ax1.set_xlabel("Time (s)")


def snip(filename, s, e, wout=True):
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
    st = int(s * 44100)
    en = int(e * 44100)
    data_s = data[st:en, :]
    if wout:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_snipped.wav", data_s, sr, "PCM_16")
    print("Done!")
    return data_s


def mix(f1, f2, r):
    n, data, data_dB, sr, ch = inputwav(f1)
    n1, data1, data_dB1, sr1, ch1 = inputwav(f2)
    data_sum = r * data + (1 - r) * data1
    sf.write(f1[0 : len(f1) - 4] + f2, data_sum, sr, "PCM_16")

    
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