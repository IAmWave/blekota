# On my machine, importing pysoundcard produces
# a lot of debugging messages. This is normal.
from pysoundcard import Stream
import soundfile
import const


def load_file(file):
    """Return an array and sampling frequency representing a sound in a given file.

    Args:
        file (str): The file to load.

    Returns:
        (array, int): A tuple containing the array of samples and the sampling rate.
        The array's values are from -1 to 1. 
    """
    return soundfile.read(file)


def save_file(file, arr, fs=const.DEFAULT_FS):
    """Save a sound given as an array and sampling frequency to a file.

    Args:
        file (str): The file to save to (does not have to exist). Should have the .wav extension.
        arr (array): The sound's values in an array with values from -1 to 1.
        fs (int, optional): Sampling frequency.
    """
    soundfile.write(file, arr, fs)


def play(arr, fs=const.DEFAULT_FS):
    """Play a sound and save it into a default file.

    Args:
        arr (array): The sound.
        fs (int, optional): Sampling frequency.
    """
    s = Stream(samplerate=fs, blocksize=16)
    s.start()
    s.write(arr)
    s.stop()
    save_file('last_played.wav', arr, fs)
