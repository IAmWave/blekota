import numpy as np


class Param:
    """An learnable parameter array.

    Attributes:
        a (array): The parameter itself; typically a matrix.
        d (array): The gradient, computed externally.
    """

    decay = 0.9
    """Coefficient for RMSprop's decaying average."""

    def __init__(self, a, name=''):
        """Initialize a parameter with given initial values and an optional name

        Args:
            a (array): The parameter's initial state.
            name (str, optional): A name to recognize the argument.
        """
        self.a = a
        self.d = np.zeros_like(a)
        # m is an exponentially decaying average of previous gradients, used for RMSprop
        self.m = np.zeros_like(a)
        self.name = name

    def step(self, alpha):
        """Adjust the parameters using RMSprop and a given learning rate.
        Assumes the gradient (self.d) has been computed.

        Args:
            alpha (float): The learning rate.
        """
        np.clip(self.d, -5, 5, out=self.d)  # clip gradients for stability
        # self.m += self.d ** 2 # AdaGrad; used in older versions
        self.m = self.decay * self.m + (1 - self.decay) * (self.d ** 2)  # RMSprop
        self.a += -alpha * self.d / np.sqrt(self.m + 1e-8)
        self.d = np.zeros_like(self.a)
