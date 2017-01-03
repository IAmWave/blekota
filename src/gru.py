import numpy as np
# import audio_io
from scipy.special import expit as sigmoid
import compander


class GRULayer:

    def __init__(self, x_n, h_n):
        coef = 0.01
        self.x_n = x_n
        self.h_n = h_n
        self.W = np.random.randn(h_n, h_n) * coef
        self.U = np.random.randn(h_n, x_n) * coef
        self.Wr = np.random.randn(h_n, h_n) * coef
        self.Ur = np.random.randn(h_n, x_n) * coef
        self.Wz = np.random.randn(h_n, h_n) * coef
        self.Uz = np.random.randn(h_n, x_n) * coef
        # self.dW, self.dU   = np.zeros_like(self.W), np.zeros_like(self.U)
        # self.dWr, self.dUr = np.zeros_like(self.Wr), np.zeros_like(self.Ur)
        # self.dWz, self.dUz = np.zeros_like(self.Wz), np.zeros_like(self.Uz)

    def forward(self, x, initial_h=None):
        if initial_h == None:
            initial_h = np.zeros(self.h_n)

        self.t_n, _ = x.shape
        self.h, self.h0, self.z, self.r, self.q = {}, {}, {}, {}, {}
        h, h0, z, r, q = self.h, self.h0, self.z, self.r, self.q
        self.x = np.copy(x)
        h[-1] = initial_h

        for t in range(self.t_n):
            z[t] = sigmoid(np.dot(self.Wz, h[t - 1]) + np.dot(self.Uz, x[t]))
            r[t] = sigmoid(np.dot(self.Wr, h[t - 1]) + np.dot(self.Ur, x[t]))
            q[t] = r[t] * h[t - 1]
            h0[t] = np.tanh(np.dot(self.W, q[t]) + np.dot(self.U, x[t]))
            h[t] = (1 - z[t]) * h[t - 1] + h0[t] * z[t]

        return h

    def backward(self, dh):
        x, h, h0, z, r, q = self.x, self.h, self.h0, self.z, self.r, self.q
        W, U, Wr, Ur, Wz, Uz = self.W, self.U, self.Wr, self.Ur, self.Wz, self.Uz
        dh0, dz, dr, dq, dx = {}, {}, {}, {}, {}
        dW, dU = np.zeros_like(W), np.zeros_like(U)
        dWr, dUr = np.zeros_like(Wr), np.zeros_like(Ur)
        dWz, dUz = np.zeros_like(Wz), np.zeros_like(Uz)

        for t in reversed(range(self.t_n)):
            dh0[t] = dh[t] * z[t]
            dh0i = dh0[t] * (1 - h0[t] ** 2)
            dW += np.dot(dh0i[:, np.newaxis], q[t][np.newaxis])  # np.outer
            dU += np.dot(dh0i[:, np.newaxis], x[t].reshape(1, -1))
            dq = np.dot(W.T, dh0i)
            # r
            dr = dq * h[t - 1]
            dri = r[t] * (1 - r[t]) * dr
            dWr += np.dot(dri[:, np.newaxis], h[t - 1][np.newaxis])
            dUr += np.dot(dri[:, np.newaxis], x[t][np.newaxis])
            # z
            dz[t] = dh[t] * (h0[t] - h[t - 1])
            dzi = z[t] * (1 - z[t]) * dz[t]
            dWz += np.dot(dzi[:, np.newaxis], h[t - 1][np.newaxis])
            dUz += np.dot(dzi[:, np.newaxis], x[t][np.newaxis])

            dx[t] = np.dot(U.T, dh0i) + np.dot(Ur.T, dri) + np.dot(Uz.T, dzi)
            if t > 0:
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + np.dot(Wr.T, dri) + np.dot(Wz.T, dzi)

        self.dW, self.dU, self.dWr, self.dUr, self.dWz, self.dUz = dW, dU, dWr, dUr, dWz, dUz

    # based on https://gist.github.com/karpathy/d4dee566867f8291f086#gistcomment-1508982
    def gradCheck(self, x):
        num_checks, delta = 10, 1e-5

        h = self.forward(x)
        # <outdated>
        h2 = {}
        for i in range(-1, 10):
            h2[i] = np.copy(h[i])

        for t in range(10):
            h2[t][t] -= 1
        # </outdated>
        self.backward(h2)

        for param, dparam, name in zip([self.W, self.U, self.Wr, self.Ur, self.Wz, self.Uz],
                                       [self.dW, self.dU, self.dWr, self.dUr, self.dWz, self.dUz],
                                       ['W', 'U', 'Wr', 'Ur', 'Wz', 'Uz']):
            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1, ('Error dims dont match: %s and %s.' % repr(s0), repr(s1))
            print(name)
            for i in range(num_checks):
                ri = int(np.random.randint(param.size))
                # evaluate cost at [x + delta] and [x - delta]
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                cg0 = cost(gru.forward(x), 10)
                param.flat[ri] = old_val - delta
                cg1 = cost(gru.forward(x), 10)
                param.flat[ri] = old_val  # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                grad_numerical = (cg0 - cg1) / (2 * delta)

                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                # rel_error should be on order of 1e-7 or less


class GRU:

    def __init__(self, h_n):
        self.x_n = 256
        self.h_n = h_n
        self.layer = GRULayer(self.x_n, h_n)
        self.Wy = np.random.randn(self.x_n, h_n) * 0.01

    def train(self, s, it=100):
        seq_length = 50
        quantized = compander.quantize(s)
        n = s.size
        x = np.zeros((n, 256))
        for i in range(n):
            x[i][quantized[i]] = 1

        p = 0
        smooth_loss = -np.log(1.0 / 256) * seq_length
        layer = self.layer
        mW, mU, mWr, mUr, mWz, mUz = [np.zeros_like(x) for x in
                                      [layer.W, layer.U, layer.Wr, layer.Ur, layer.Wz, layer.Uz]]

        for i in range(it):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= n or i == 0:
                h = np.zeros(layer.h_n)  # reset RNN memory
                p = 0  # go from start of data
            inputs = x[p:p + seq_length]
            targets = x[p + 1:p + seq_length + 1]

            h = layer.forward(inputs)
            dh = {}
            loss = 0
            for j in range(seq_length):
                dh[j] = np.exp(h[j]) / np.sum(np.exp(h[j]))

                loss += -np.log(dh[j][quantized[j]])
                dh[j][quantized[j]] -= 1

            layer.backward(dh)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if i % 1 == 0:
                print('iter:\t%d\tloss:\t%f' % (i, loss))  # print progress

            alpha = 0.03
            for param, dparam, mem in zip([layer.W, layer.U, layer.Wr, layer.Ur, layer.Wz, layer.Uz],
                                          [layer.dW, layer.dU, layer.dWr, layer.dUr, layer.dWz, layer.dUz],
                                          [mW, mU, mWr, mUr, mWz, mUz]):
                mem += dparam * dparam
                param += -alpha * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += seq_length  # move data pointer

    """
    def sample(self, n, hint=np.zeros(1)):
        x = np.zeros((n, 255))
        x[0][compander.quantize(hint[0])] = 1
        return np.random.randn(n)
    """

    def sample(self, n, hint=np.zeros(1)):
        l = self.layer
        h = np.zeros(l.h_n)
        x = np.zeros(256)
        x[127] = 0
        y = np.zeros(n)

        for t in range(n):
            z = sigmoid(np.dot(l.Wz, h) + np.dot(l.Uz, x))
            r = sigmoid(np.dot(l.Wr, h) + np.dot(l.Ur, x))
            q = r * h
            h0 = np.tanh(np.dot(l.W, q) + np.dot(l.U, x))
            h = (1 - z) * h + h0 * z
            p = np.exp(h) / np.sum(np.exp(h))
            chosen = np.random.choice(range(256), p=p.ravel())
            x = np.zeros(256)
            x[chosen] = 1
            y[t] = chosen

        return compander.unquantize(y)


"""
def cost(h, n):
    err = 0
    for t in range(n):
        a = np.copy(h[t])
        a[t] -= 1
        err += np.sum(np.square(a))
    return err / 2

# y, fs = audio_io.load_file('data/shapes/sin_440Hz_8kHz_3s.wav')
# y = y[0:10]
gru = GRULayer(1, 10)
# gru.gradCheck(np.arange(1, 11)[:,None])


for it in range(1000):
    y = np.arange(1, 11)[:,None]
    h = gru.forward(y)

    print(cost(h, 10))
    # print(h[5])
    for t in range(10):
        h[t][t] -= 1
    gru.backward(h)


"""
