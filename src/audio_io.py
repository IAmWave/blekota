from pysoundcard import Stream
import soundfile
import const

# load_file returns (arr, fs), where
# arr = numpy array with values from -1 to 1
# fs = sample rate
from soundfile import read as load_file

# file name, data array, sample rate
from soundfile import write as save_file


def play(arr, fs=const.DEFAULT_FS):
    s = Stream(samplerate=fs, blocksize=16)
    s.start()
    s.write(arr)
    s.stop()
    save_file('foo.wav', arr, fs)
