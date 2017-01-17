import numpy as np
from scipy.special import expit as sigmoid


class GRULayer:

    def __init__(self, x_n, h_n):
        coef = 0.01
        self.x_n = x_n
        self.h_n = h_n
        self.W = np.random.randn(h_n, x_n + h_n + 1) * coef
        self.Wr = np.random.randn(h_n, x_n + h_n + 1) * coef
        self.Wz = np.random.randn(h_n, x_n + h_n + 1) * coef

    def forward(self, x, initial_h=None):
        if initial_h is None:
            initial_h = np.zeros(self.h_n)

        self.t_n, _ = x.shape
        self.h = np.zeros((self.t_n, self.h_n))
        self.h0, self.z, self.r, self.q = np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h)
        h, h0, z, r, q = self.h, self.h0, self.z, self.r, self.q
        self.x = np.copy(x)

        for t in range(self.t_n):
            h_prev = h[t - 1] if t > 0 else initial_h
            z[t] = sigmoid(np.dot(self.Wz, np.r_[x[t], h_prev, np.ones(1)]))
            r[t] = sigmoid(np.dot(self.Wr, np.r_[x[t], h_prev, np.ones(1)]))
            q[t] = r[t] * h_prev
            h0[t] = np.tanh(np.dot(self.W, np.r_[x[t], q[t], np.ones(1)]))
            h[t] = (1 - z[t]) * h_prev + h0[t] * z[t]

        return h

    def backward(self, dh):
        x, h, h0, z, r, q = self.x, self.h, self.h0, self.z, self.r, self.q
        W, Wr, Wz, = self.W, self.Wr, self.Wz
        dh0, dz, dr, dq, dx = np.zeros_like(dh), np.zeros_like(dh), np.zeros_like(dh), np.zeros_like(dh), np.zeros((self.t_n, self.x_n))
        dW, dWr, dWz = np.zeros_like(W), np.zeros_like(Wr), np.zeros_like(Wz)

        for t in reversed(range(self.t_n)):
            h_prev = h[t - 1] if t > 0 else np.zeros(self.h_n)
            dh0[t] = dh[t] * z[t]
            dh0i = dh0[t] * (1 - h0[t] ** 2)
            dW += np.dot(dh0i[:, np.newaxis], np.r_[x[t], q[t], np.ones(1)][None])
            dxq = np.dot(W.T, dh0i)
            dx[t] += dxq[:self.x_n]
            dq = dxq[self.x_n:-1]
            # r
            dr = dq * h_prev
            dri = r[t] * (1 - r[t]) * dr
            dWr += np.dot(dri[:, None], np.r_[x[t], h_prev, np.ones(1)][None])
            dxh = np.dot(Wr.T, dri)
            # z
            dz[t] = dh[t] * (h0[t] - h_prev)
            dzi = z[t] * (1 - z[t]) * dz[t]
            dWz += np.dot(dzi[:, None], np.r_[x[t], h_prev, np.ones(1)][None])
            dxh += np.dot(Wz.T, dzi)
            dx[t] += dxh[:self.x_n]
            if t > 0:
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + dxh[self.x_n:-1]

        self.dW, self.dWr, self.dWz = dW, dWr, dWz
