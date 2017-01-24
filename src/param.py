import numpy as np


class Param:

    def __init__(self, a, name=''):
        self.a = a
        self.d = np.zeros_like(a)
        self.m = np.zeros_like(a)
        self.name = name

    # adagrad update
    def step(self, alpha):
        np.clip(self.d, -5, 5, out=self.d)
        self.m += self.d ** 2
        self.a += -alpha * self.d / np.sqrt(self.m + 1e-8)
        self.d = np.zeros_like(self.a)
