import soundfile as sf
import time
from utilities import inputwav
import numpy as np


def compress(filename, threshold, ratio, makeup, attack, release, wout=True):
    """
    Reduces dynamic range of input signal by reducing volume above threshold.
    The gain reduction is smoothened according to the attack and release.
    Makeup gain must be added manually.
    Parameters
    ----------
    filename: string
        Name of the input *.wav file.
    threshold: scalar (dB)
        value in dB of the threshold at which the compressor engages in
        gain reduction.
    ratio: scalar
        The ratio at which volume is reduced for every dB above the threshold
        (i.e. r:1)
        For compression to occur, ratio should be above 1.0. Below 1.0, you
        are expanding the signal.

    makeup: scalar (dB)
        Amount of makeup gain to apply to the compressed signal

    attack: scalar (ms)
        Characteristic time required for compressor to apply full gain
        reduction. Longer times allow transients to pass through while short
        times reduce all of the signal. Distortion will occur if the attack
        time is too short.

    release: scalar (ms)
        Characteristic time that the compressor will hold the gain reduction
        before easing off. Both attack and release basically smoothen the gain
        reduction curves.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.



    Returns
    -------
    data_Cs: array containing the compressed waveform in dB
    data_Cs_bit: array containing the compressed waveform in bits.
    """
    start = time.time()
    if ratio < 1.0:
        print("Ratio must be > 1.0 for compression to occur! You are expanding.")
    if ratio == 1.0:
        print("Signal is unaffected.")
    n, data, data_dB, sr, ch = inputwav(filename)
    # Array for the compressed data in dB
    dataC = data_dB.copy()
    # attack and release time constant
    a = np.exp(-np.log10(9) / (44100 * attack * 1.0e-3))
    re = np.exp(-np.log10(9) / (44100 * release * 1.0e-3))
    # apply compression
    print("Compressing...")
    for k in range(ch):
        for i in range(n):
            if dataC[i, k] > threshold:
                dataC[i, k] = threshold + (dataC[i, k] - threshold) / (ratio)
    # gain and smooth gain initialization
    gain = np.zeros(n)
    sgain = np.zeros(n)
    # calculate gain
    gain = np.subtract(dataC, data_dB)
    sgain = gain.copy()
    # smoothen gain
    print("Smoothing...")
    for k in range(ch):
        for i in range(1, n):
            if sgain[i - 1, k] >= sgain[i, k]:
                sgain[i, k] = a * sgain[i - 1, k] + (1 - a) * sgain[i, k]
            if sgain[i - 1, k] < sgain[i, k]:
                sgain[i, k] = re * sgain[i - 1, k] + (1 - re) * sgain[i, k]
    # Array for the smooth compressed data with makeup gain applied
    dataCs = np.zeros(n)
    dataCs = data_dB + sgain + makeup
    # Convert our dB data back to bits
    dataCs_bit = 10.0 ** ((dataCs) / 20.0)
    # sign the bits appropriately:
    for k in range(ch):
        for i in range(n):
            if data[i, k] < 0.0:
                dataCs_bit[i, k] = -1.0 * dataCs_bit[i, k]
    # write data to 16 bit file
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_compressed.wav",
            dataCs_bit,
            sr,
            "PCM_16",
        )
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("Done!")
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return dataCs, dataCs_bit


def limit(filename, threshold, makeup, wout=True):
    """
    Limits the data above the threshold. Compression with high ratio and long
    release with fairly quick attack.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    threshold : scalar (dB)
        value in dB of the threshold above which signal is limited

    makeup: scalar (dB)
        Amount of makeup gain to apply to the compressed signal

    attack: scalar (ms)
        Characteristic time required for compressor to apply full gain
        reduction. Longer times allow transients to pass through while short
        times reduce all of the signal. Distortion will occur if the attack
        time is too short.

    release: scalar (ms)
        Characteristic time that the compressor will hold the gain reduction
        before easing off. Both attack and release basically smoothen the gain
        reduction curves.

    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.



    Returns
    -------
    dataL: array containing the limited waveform in dB

    dataL_bit: array containing the limited waveform in bits.
    """
    start = time.time()
    n, data, data_dB, sr, ch = inputwav(filename)
    dataL, dataL_bit = compress(
        filename, threshold, 1000.0, makeup, 1.0, 500.0, wout=False
    )
    if wout:
        print("Exporting...")
        sf.write(
            filename[0 : len(filename) - 4] + "_limit.wav", dataL_bit, 44100, "PCM_16"
        )
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("Done!")
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return dataL, dataL_bit


def distort(filename, threshold=0.25, type="arctan", wout=True):
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
    if wout:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_distort.wav", dataD, sr, "PCM_16")

    end = time.time()
    elapsed = int(1000 * (end - start))
    print("Completed in " + str(elapsed) + " milliseconds.")
    return dataD
