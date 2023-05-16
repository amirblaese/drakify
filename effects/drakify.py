from dynamic_effects import *
from filters import *
from reverb import *
from time_effects import *

import os


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
