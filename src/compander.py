# uses the μ-law algorithm

import numpy as np

mu = 255

compress = np.vectorize(
    lambda x: np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu)))

expand = np.vectorize(
    lambda y: np.sign(y) * ((1 + mu)**(np.abs(y)) - 1) / mu)


def quantize(x):
    return np.floor((compress(x) + 1) * 0.5 * mu).astype(int)


def unquantize(y):
    return expand((y * 2 / mu) - 1)

"""
def expand1(y):
    return np.sign(y) * ((1 + mu)**(np.abs(y)) - 1) / mu

def compress1(x):
    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
"""
