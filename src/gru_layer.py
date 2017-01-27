import numpy as np
from scipy.special import expit as sigmoid
from param import Param


class GRULayer:

    def __init__(self, x_n, h_n, name=''):
        coef = 0.01
        self.x_n = x_n
        self.h_n = h_n
        self.W = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_W'))
        self.Wr = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_Wr'))
        self.Wz = Param(np.random.randn(h_n, x_n + h_n + 1) * coef, name=(name + ' GRU_Wz'))

    def forward(self, x, initial_h=None):
        # x:          (seq_length, x_n (#features), batch_size)
        # initial_h:  (h_n, batch_size)
        self.t_n, _, self.b = x.shape
        b = self.b

        if initial_h is None:
            initial_h = np.zeros((self.h_n, b))

        self.h = np.zeros((self.t_n, self.h_n, b))

        self.h0, self.z, self.r, self.q = [np.zeros_like(self.h) for x in range(4)]

        h, h0, z, r, q = self.h, self.h0, self.z, self.r, self.q

        self.x = np.copy(x)
        for t in range(self.t_n):
            h_prev = h[t - 1] if t > 0 else initial_h

            z[t] = sigmoid(np.dot(self.Wz.a, np.r_[x[t], h_prev, np.ones((1, b))]))
            r[t] = sigmoid(np.dot(self.Wr.a, np.r_[x[t], h_prev, np.ones((1, b))]))
            q[t] = r[t] * h_prev
            h0[t] = np.tanh(np.dot(self.W.a, np.r_[x[t], q[t], np.ones((1, b))]))
            h[t] = (1 - z[t]) * h_prev + h0[t] * z[t]

        return np.copy(h)

    def backward(self, dh):
        # shape of dh should be (t_n, h_n, b) = (seq_length, hidden_size, batch_size)
        x, h, h0, z, r, q, b = self.x, self.h, self.h0, self.z, self.r, self.q, self.b
        W, Wr, Wz = self.W, self.Wr, self.Wz

        dh0, dz, dr, dq = [np.zeros_like(dh) for x in range(4)]
        dx = np.zeros((self.t_n, self.x_n, b))

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
            if t > 0:
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + dxh[self.x_n:-1]

        return np.copy(dx)

    def getParams(self):
        return [self.W, self.Wr, self.Wz]

# based on https://gist.github.com/karpathy/d4dee566867f8291f086#gistcomment-1508982


def gradCheck(l=GRULayer(1, 10)):

    def cost(h):
        dh = h - np.linspace(-1, 1, h.shape[0])[:, None, None]
        return 0.5 * np.sum(dh * dh), dh

    num_checks, delta = 5, 1e-5
    n = 20
    x = np.arange(n * 2.0).reshape((n, 1, 2))
    h = l.forward(x)
    dh = cost(h)[1]

    dx = l.backward(dh)

    for param, name in zip([x, l.W, l.Wr, l.Wz],
                           ['x', 'W', 'Wr', 'Wz']):

        print(name)
        a = param if (name == 'x') else param.a

        for i in range(num_checks):
            ri = int(np.random.randint(a.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = a.flat[ri]
            a.flat[ri] = old_val + delta
            cg0 = cost(l.forward(x))[0]
            a.flat[ri] = old_val - delta
            cg1 = cost(l.forward(x))[0]
            a.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = (param.d if (name != 'x') else dx).flat[ri]

            grad_numerical = (cg0 - cg1) / (2 * delta)

            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less

# gradCheck()
