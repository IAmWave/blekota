import numpy as np
# import audio_io
from scipy.special import expit as sigmoid
import compander


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
            z[t] = sigmoid(np.dot(self.Wz, np.concatenate((x[t], h_prev, np.ones(1)))))
            r[t] = sigmoid(np.dot(self.Wr, np.concatenate((x[t], h_prev, np.ones(1)))))
            q[t] = r[t] * h_prev
            h0[t] = np.tanh(np.dot(self.W, np.concatenate((x[t], q[t], np.ones(1)))))
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
            dW += np.dot(dh0i[:, np.newaxis], np.concatenate((x[t], q[t], np.ones(1)))[None])
            dxq = np.dot(W.T, dh0i)
            dx[t] += dxq[:self.x_n]
            dq = dxq[self.x_n:-1]
            # r
            dr = dq * h_prev
            dri = r[t] * (1 - r[t]) * dr
            dWr += np.dot(dri[:, None], np.concatenate((x[t], h_prev, np.ones(1)))[None])
            dxh = np.dot(Wr.T, dri)
            # z
            dz[t] = dh[t] * (h0[t] - h_prev)
            dzi = z[t] * (1 - z[t]) * dz[t]
            dWz += np.dot(dzi[:, None], np.concatenate((x[t], h_prev, np.ones(1)))[None])
            dxh += np.dot(Wz.T, dzi)
            dx[t] += dxh[:self.x_n]
            if t > 0:
                dh[t - 1] += dh[t] * (1 - z[t]) + dq * r[t] + dxh[self.x_n:-1]

        self.dW, self.dWr, self.dWz = dW, dWr, dWz


class GRU:

    def __init__(self, h_n):
        self.x_n = 256
        self.h_n = h_n
        self.layer = GRULayer(self.x_n, h_n)
        self.Wy = np.random.randn(self.x_n, h_n) * 0.01
        self.by = np.zeros(self.x_n)
        self.mW, self.mWr, self.mWz, self.mWy, self.mby = \
            [np.zeros_like(x) for x in [self.layer.W, self.layer.Wr, self.layer.Wz, self.Wy, self.by]]

    def train(self, s, it=100):
        seq_length = 90
        loss_report_n = 100

        quantized = compander.quantize(s)
        n = s.size
        x = np.zeros((n, 256))
        for i in range(n):
            x[i][quantized[i]] = 1

        p = 0
        layer = self.layer
        losses = np.zeros(loss_report_n)

        for i in range(it):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= n or i == 0:
                h_prev = np.zeros(layer.h_n)  # reset RNN memory
                p = 0  # go from start of data
                print('New epoch')
            inputs = x[p:p + seq_length]
            #targets = x[p + 1:p + seq_length + 1]

            h = layer.forward(inputs, initial_h=h_prev)
            dh = np.zeros_like(h)
            loss = 0
            dWy = np.zeros_like(self.Wy)
            dby = np.zeros_like(self.by)
            for j in range(seq_length):
                y = np.dot(self.Wy, h[j][:, np.newaxis]).flatten() + self.by
                if np.max(y) > 500:
                    print('Warning: y value too large: ', np.max(y))
                    print(np.dot(self.Wy, np.ones(500, 1)))
                    print(self.by)
                    print('Iteration ', i)
                    return
                    np.clip(y, None, 500, out=y)

                dy = np.exp(y) / np.sum(np.exp(y))  # also means the probabilities
                loss += -np.log(dy[quantized[p + 1 + j]])
                dy[quantized[p + 1 + j]] -= 1

                dby += dy
                dWy += np.dot(dy[:, np.newaxis], h[j][np.newaxis])
                dh[j] = np.dot(self.Wy.T, dy[:, np.newaxis]).flatten()

            h_prev = np.copy(h[-1])
            layer.backward(dh)

            losses[i % loss_report_n] = loss
            if i % loss_report_n == (loss_report_n - 1):
                print(losses)
                print('iter:\t%d\tloss:\t%f' % (i + 1, np.mean(losses)))  # print progress

            alpha = 1e-2
            for param, dparam, mem in zip([layer.W, layer.Wr, layer.Wz, self.Wy, self.by],
                                          [layer.dW, layer.dWr, layer.dWz, dWy, dby],
                                          [self.mW, self.mWr, self.mWz, self.mWy, self.mby]):
                np.clip(dparam, -5, 5, out=dparam)
                mem += dparam * dparam
                param += -alpha * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += seq_length  # move data pointer

    def sample(self, n, hint=np.zeros(1)):
        l = self.layer
        h = np.zeros(l.h_n)
        x = np.zeros(256)
        x[compander.quantize(hint[0])] = 0
        res = np.zeros(n)
        p = np.zeros((n, 256))

        for t in range(n):
            """
            z = sigmoid(np.dot(l.Wz, np.concatenate((x, h, np.ones(1)))))
            r = sigmoid(np.dot(l.Wr, np.concatenate((x, h, np.ones(1)))))
            q = r * h
            h0 = np.tanh(np.dot(l.W, np.concatenate((x, q, np.ones(1)))))
            h = (1 - z) * h + h0 * z
            y = np.dot(self.Wy, h[:, None]).flatten() + self.by
            """

            h = self.layer.forward(x[None], initial_h=h).flatten()
            y = np.dot(self.Wy, h[:, None]).flatten() + self.by

            p[t] = np.exp(y) / np.sum(np.exp(y))
            chosen = np.random.choice(range(256), p=p[t].ravel())
            x = np.zeros(256)
            if t < hint.shape[0]:
                x[compander.quantize(hint[t])] = 1
            else:
                x[chosen] = 1
            res[t] = chosen

        self.p = p
        return compander.unquantize(res)


# based on https://gist.github.com/karpathy/d4dee566867f8291f086#gistcomment-1508982
def gradCheck(l=GRULayer(1, 10)):

    def cost(h):
        dh = h - np.linspace(-1, 1, h.shape[0])[:, None]
        return 0.5 * np.sum(dh * dh)

    num_checks, delta = 5, 1e-5
    n = 20
    x = np.arange(n)[:, None]

    h = l.forward(x)
    dh = h - np.linspace(-1, 1, n)[:, None]

    l.backward(dh)

    for param, dparam, name in zip([l.W, l.Wr, l.Wz],
                                   [l.dW, l.dWr, l.dWz],
                                   ['W', 'Wr', 'Wz']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, ('Error dims dont match: %s and %s.' % repr(s0), repr(s1))
        print(name)

        for i in range(num_checks):
            ri = int(np.random.randint(param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0 = cost(l.forward(x))
            param.flat[ri] = old_val - delta
            cg1 = cost(l.forward(x))
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)

            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less
