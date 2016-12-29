from pysoundcard import Stream
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import numpy as np
import sounddevice as sd
import const


def load_file(name):
    fs, arr = read_wav(name)
    arr = arr / (np.iinfo(arr.dtype).max + 1)  # -1 to 1
    return arr, fs

def quantize(x, mu):
    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))


def play(arr, fs=const.DEFAULT_FS, blocking=False):
    blocksize = 16
    s = Stream(samplerate=fs, blocksize=blocksize)
    s.start()
    s.write(arr)
    s.stop()
    write_wav('foo.wav', fs, arr)
    #sd.play(arr, fs, blocking=blocking)