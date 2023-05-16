from utilities import inputwav
import time
import numpy as np
from scipy.signal import convolve
import soundfile as sf


def conv_reverb(filename, ir="/../assets/default_reverb_IR.wav", wet=0.2, wout=True):
    start = time.time()
    print(filename)
    n, data, data_dB, sr, ch = inputwav(filename)
    n_IR, data_IR, data_dB_IR, sr_IR, ch_IR = inputwav(ir)
    print("Convolving impulse response with track...")
    convolution = convolve(data, data_IR, mode="full")
    for i in range(ch + 1):  # normalize the song
        convolution[:, i] = convolution[:, i] / max(convolution[:, i])
    song_pad = np.zeros(convolution[:, 0:2].shape)
    for i in range(ch):
        song_pad[0 : len(data)] = data
    print("Mixing...")
    final = (1 - wet) * song_pad + (wet) * convolution[:, 0:2]
    if wout:
        print("Exporting...")
        sf.write(filename[0 : len(filename) - 4] + "_verbed.wav", final, sr, "PCM_16")
    print("Done!")
    end = time.time()
    elapsed = int(1000 * (end - start))
    print("...............................")
    print("Completed in " + str(elapsed) + " milliseconds.")
    return final
