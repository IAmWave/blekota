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
        self.br = np.zeros(h_n)
        self.bz = np.zeros(h_n)
        self.bh = np.zeros(h_n)

    def forward(self, x, initial_h=None):
        if initial_h == None:
            initial_h = np.zeros(self.h_n)

        self.t_n, _ = x.shape
        self.h = np.zeros((self.t_n, self.h_n))
        self.h0, self.z, self.r, self.q = np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h)
        h, h0, z, r, q = self.h, self.h0, self.z, self.r, self.q
        self.x = np.copy(x)

        for t in range(self.t_n):
            prev_h = h[t - 1] if t > 0 else initial_h
            z[t] = sigmoid(np.dot(self.Wz, prev_h) + np.dot(self.Uz, x[t]) + self.bz)
            r[t] = sigmoid(np.dot(self.Wr, prev_h) + np.dot(self.Ur, x[t]) + self.br)
            q[t] = r[t] * prev_h
            h0[t] = np.tanh(np.dot(self.W, q[t]) + np.dot(self.U, x[t]) + self.bh)
            h[t] = (1 - z[t]) * prev_h + h0[t] * z[t]

        return h

    def backward(self, dh):
        x, h, h0, z, r, q = self.x, self.h, self.h0, self.z, self.r, self.q
        W, U, Wr, Ur, Wz, Uz, br, bz, bh = self.W, self.U, self.Wr, self.Ur, self.Wz, self.Uz, self.br, self.bz, self.bh
        dh0, dz, dr, dq, dx = np.zeros_like(dh), np.zeros_like(dh), np.zeros_like(dh), np.zeros_like(dh), np.zeros((self.t_n, self.x_n))
        dW, dU, dWr, dUr, dWz, dUz, dbr, dbz, dbh = [np.zeros_like(param) for param in
                                                     [W, U, Wr, Ur, Wz, Uz, br, bz, bh]]

        for t in reversed(range(self.t_n)):
            dh0[t] = dh[t] * z[t]
            dh0i = dh0[t] * (1 - h0[t] ** 2)
            dW += np.dot(dh0i[:, np.newaxis], q[t][np.newaxis])  # np.outer
            dU += np.dot(dh0i[:, np.newaxis], x[t].reshape(1, -1))
            dbh += dh0i
            dq = np.dot(W.T, dh0i)
            # r
            dr = dq * h[t - 1]
            dri = r[t] * (1 - r[t]) * dr
            dWr += np.dot(dri[:, np.newaxis], h[t - 1][np.newaxis])
            dUr += np.dot(dri[:, np.newaxis], x[t][np.newaxis])
            dbr += dri
            # z
            dz[t] = dh[t] * (h0[t] - h[t - 1])
            dzi = z[t] * (1 - z[t]) * dz[t]
            dWz += np.dot(dzi[:, np.newaxis], h[t - 1][np.newaxis])
            dUz += np.dot(dzi[:, np.newaxis], x[t][np.newaxis])
            dbz += dzi

            dx[t] = np.dot(U.T, dh0i) + np.dot(Ur.T, dri) + np.dot(Uz.T, dzi)
            if t > 0:
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + np.dot(Wr.T, dri) + np.dot(Wz.T, dzi)

        self.dW, self.dU, self.dWr, self.dUr, self.dWz, self.dUz, self.dbr, self.dbz, self.dbh \
            = dW, dU, dWr, dUr, dWz, dUz, dbr, dbz, dbh

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
        self.by = np.zeros(self.x_n)

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
        mW, mU, mWr, mUr, mWz, mUz, mbr, mbz, mbh, mWy, mby = [np.zeros_like(x) for x in
                                                               [layer.W, layer.U, layer.Wr, layer.Ur, layer.Wz, layer.Uz,
                                                                layer.br, layer.bz, layer.bh, self.Wy, self.by]]

        for i in range(it):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= n or i == 0:
                h = np.zeros(layer.h_n)  # reset RNN memory
                p = 0  # go from start of data
            inputs = x[p:p + seq_length]
            targets = x[p + 1:p + seq_length + 1]

            h = layer.forward(inputs)
            dh = np.zeros_like(h)
            loss = 0
            dWy = np.zeros_like(self.Wy)
            dby = np.zeros_like(self.by)
            for j in range(seq_length):
                y = np.dot(self.Wy, h[j][:, np.newaxis]).flatten() + self.by
                dy = np.exp(y) / np.sum(np.exp(y))  # also means the probabilities
                loss += -np.log(dy[quantized[j]])
                dy[quantized[j]] -= 1

                dby += dy
                dWy += np.dot(dy[:, np.newaxis], h[j][np.newaxis])
                dh[j] = np.dot(self.Wy.T, dy[:, np.newaxis]).flatten()

                #dh[j] = np.exp(h[j]) / np.sum(np.exp(h[j]))
                #loss += -np.log(dh[j][quantized[j]])
                #dh[j][quantized[j]] -= 1

            layer.backward(dh)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if i % 100 == 0:
                print('iter:\t%d\tloss:\t%f' % (i, smooth_loss))  # print progress

            alpha = 0.01
            for param, dparam, mem in zip([layer.W, layer.U, layer.Wr, layer.Ur, layer.Wz, layer.Uz, layer.br, layer.bz, layer.bh, self.Wy, self.by],
                                          [layer.dW, layer.dU, layer.dWr, layer.dUr, layer.dWz, layer.dUz, layer.dbr, layer.dbz, layer.dbh, dWy, dby],
                                          [mW, mU, mWr, mUr, mWz, mUz, mbr, mbz, mbh, mWy, mby]):
                mem += dparam * dparam
                param += -alpha * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += seq_length  # move data pointer

    def sample(self, n, hint=np.zeros(1)):
        l = self.layer
        h = np.zeros(l.h_n)
        x = np.zeros(256)
        x[compander.quantize(hint[0])] = 0
        res = np.zeros(n)

        for t in range(n):
            z = sigmoid(np.dot(l.Wz, h) + np.dot(l.Uz, x))
            r = sigmoid(np.dot(l.Wr, h) + np.dot(l.Ur, x))
            q = r * h
            h0 = np.tanh(np.dot(l.W, q) + np.dot(l.U, x))
            h = (1 - z) * h + h0 * z
            y = np.dot(self.Wy, h[:, np.newaxis]).flatten() + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            chosen = np.random.choice(range(256), p=p.ravel())
            x = np.zeros(256)
            x[chosen] = 1
            res[t] = chosen

        return compander.unquantize(res)
