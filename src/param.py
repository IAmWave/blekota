import numpy as np


class Param:

    decay = 0.95

    def __init__(self, a, name=''):
        self.a = a
        self.d = np.zeros_like(a)
        self.m = np.zeros_like(a)
        self.name = name

    def step(self, alpha):
        np.clip(self.d, -5, 5, out=self.d)
        # self.m += self.d ** 2 # adagrad
        self.m = self.decay * self.m + (1 - self.decay) * (self.d ** 2)  # rmsprop
        self.a += -alpha * self.d / np.sqrt(self.m + 1e-8)
        self.d = np.zeros_like(self.a)
