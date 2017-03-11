import numpy as np
from param import Param


def sigmoid(x):
    """Calculate the element-wise logistic sigmoid function of the input array."""
    return np.reciprocal(1 + np.exp(-x))


class GRULayer:
    """A single GRU layer (unit), part of a larger GRU (see gru.py).

    Very loosely based on Andrej Karpathy's code: https://gist.github.com/karpathy/d4dee566867f8291f086

    Attributes:
        x_n (int): Input size.
        h_n (int): Hidden layer (output) size.
        W, Wr, Wz (Params): Learnable parameters (matrices).
    """

    def __init__(self, x_n, h_n, name=''):
        """Initializes a GRULayer with a specified input/output size and an optional name."""
        self.x_n = x_n
        self.h_n = h_n
        coef = 0.01  # Small default weights (parameter values)
        self.W = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_W'))
        self.Wr = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_Wr'))
        self.Wz = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_Wz'))

    def forward(self, x, initial_h=None):
        """Perform several forward passes of the layer, computing the hidden states at each step.
        The forward pass is batched, meaning it can works on multiple sequences simultaneously
        We denote the number of sequences by b (meaning batch size) and their length by t_n.

        Args:
            x (array): Input array of shape (t_n, x_n, b).
            initial_h (array, optional): The first hidden state. Shape (h_n, b).

        Returns:
            h (array): Array of shape (t_n, h_n, b) - the computed hidden states.
        """

        self.t_n, _, self.b = x.shape  # determine sequence length and batch size from input
        b = self.b

        if initial_h is None:
            initial_h = np.zeros((self.h_n, b))

        self.h = np.zeros((self.t_n, self.h_n, b))

        self.h0, self.z, self.r, self.q = [np.zeros_like(self.h) for x in range(4)]

        # less "self" everywhere
        h, h0, z, r, q = self.h, self.h0, self.z, self.r, self.q

        self.x = np.copy(x)
        for t in range(self.t_n):
            h_prev = h[t - 1] if t > 0 else initial_h
            # GRU step; Cho et al. 2014 (arXiv:1406.1078v3) for explanation
            z[t] = sigmoid(np.dot(self.Wz.a, np.r_[x[t], h_prev, np.ones((1, b))]))
            r[t] = sigmoid(np.dot(self.Wr.a, np.r_[x[t], h_prev, np.ones((1, b))]))
            q[t] = r[t] * h_prev
            h0[t] = np.tanh(np.dot(self.W.a, np.r_[x[t], q[t], np.ones((1, b))]))
            h[t] = (1 - z[t]) * h_prev + h0[t] * z[t]

        return np.copy(h)

    def backward(self, dh):
        """Perform several backward passes of the layer, computing the gradients of all parameters and inputs.
        Assumes a forward pass was peformed before.

        Args:
            dh (array): The gradient of h in an array of shape (t_n, h_n, b) - see forward() for details

        Returns:
            dx (array): The gradient of x, the original input from forward()
        """

        # less "self" everywhere
        x, h, h0, z, r, q, b = self.x, self.h, self.h0, self.z, self.r, self.q, self.b
        W, Wr, Wz = self.W, self.Wr, self.Wz

        dh0, dz, dr, dq = [np.zeros_like(dh) for x in range(4)]
        dx = np.zeros((self.t_n, self.x_n, b))

        # The formulas for the individual derivatives are fairly complex and not necessary
        # for understanding the model; I leave the code mostly uncommented here
        for t in reversed(range(self.t_n)):
            h_prev = h[t - 1] if t > 0 else np.zeros((self.h_n, b))
            dh0[t] = dh[t] * z[t]
            dh0i = dh0[t] * (1 - h0[t] ** 2)
            W.d += np.dot(dh0i, np.r_[x[t], q[t], np.ones((1, b))].T)
            dxq = np.dot(W.a.T, dh0i)
            dx[t] += dxq[:self.x_n]
            dq = dxq[self.x_n:-1]
            # r
            dr = dq * h_prev
            dri = r[t] * (1 - r[t]) * dr
            Wr.d += np.dot(dri, np.r_[x[t], h_prev, np.ones((1, b))].T)
            dxh = np.dot(Wr.a.T, dri)
            # z
            dz[t] = dh[t] * (h0[t] - h_prev)
            dzi = z[t] * (1 - z[t]) * dz[t]
            Wz.d += np.dot(dzi, np.r_[x[t], h_prev, np.ones((1, b))].T)
            dxh += np.dot(Wz.a.T, dzi)
            dx[t] += dxh[:self.x_n]
            if t > 0:  # h[t-1] affected h[t], update gradients to reflect this
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + dxh[self.x_n:-1]

        return dx

    def getParams(self):
        """Return a list of the model's parameters (Param objects)"""
        return [self.W, self.Wr, self.Wz]

    def __getstate__(self):
        """Return a representation of the object for pickling - only pickle layer parameters (omitting temporary computed values)."""
        state = {
            'x_n': self.x_n,
            'h_n': self.h_n,
            'W': self.W,
            'Wr': self.Wr,
            'Wz': self.Wz
        }
        return state


def gradCheck(l=GRULayer(1, 10)):
    """Gradient check - determine whether the computed (analytical) gradient matches the numberical one (for debugging).

    Checks the gradients of the parameters as well as the input(necessary for multilayer networks).
    Based on Andrej Karpathy's(@karpathy) code.
    https: // gist.github.com / karpathy / d4dee566867f8291f086  # gistcomment-1508982

    Args:
        l(GRULayer, optional): The GRULayer to check.
    """

    def loss(h):
        """A dummy loss function; the square error compared to a linspace."""
        dh = h - np.linspace(-1, 1, h.shape[0])[:, None, None]
        return 0.5 * np.sum(dh * dh), dh

    num_checks = 5
    delta = 1e-5
    n = 20
    x = np.arange(n * 2.0).reshape((n, 1, 2))  # dummy input; batch of size 2, 20 samples per sequence
    h = l.forward(x)
    dh = loss(h)[1]
    dx = l.backward(dh)  # analytical gradient

    for param, name in zip([x, l.W, l.Wr, l.Wz],
                           ['x', 'W', 'Wr', 'Wz']):

        print(name)
        a = param if (name == 'x') else param.a  # only x is not a Param object

        for i in range(num_checks):
            ri = int(np.random.randint(a.size))
            # compute the derivative from definition - evaluate loss at [x+delta] and [x-delta]
            old_val = a.flat[ri]
            a.flat[ri] = old_val + delta
            cg0 = loss(l.forward(x))[0]
            a.flat[ri] = old_val - delta
            cg1 = loss(l.forward(x))[0]
            a.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = (dx if (name == 'x') else param.d).flat[ri]  # again, treat x differently
            grad_numerical = (cg0 - cg1) / (2 * delta)

            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less
