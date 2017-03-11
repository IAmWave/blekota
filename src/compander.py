"""Implements the μ-law algorithm as in WaveNet: https://arxiv.org/pdf/1609.03499.pdf"""

import numpy as np
import const

mu = 255
"""Standard value of μ, a constant used in the algorithm; determines the "zooom" of the ln function"""


def quantize(x):
    """Quantize sound's samples to integers from 0 to 255 using the μ-law compander.

    Args:
        x (array): The sound array with float values from -1 to 1.

    Returns:
        The quantized array with integer values from 0 to 255.
    """
    return np.floor((compress(x) + 1) * 0.5 * const.SAMPLE_VALUES).astype(int)


def unquantize(y):
    """The inverse function of quantize. Approximates the original float values from a quantized array.

    Args:
        x (array): A quantized array (with values from 0 to 255).

    Returns:
        Array approximating the original array before quantization.
    """
    return expand((y * 2 / mu) - 1)


compress = np.vectorize(
    lambda x: np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu)))
"""Compress a given array using μ-law (helper function for quantize)"""

expand = np.vectorize(
    lambda y: np.sign(y) * ((1 + mu)**(np.abs(y)) - 1) / mu)
"""Expand a given array using μ-law (helper function for unquantize)"""
