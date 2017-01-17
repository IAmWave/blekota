import numpy as np
import compander
from gru_layer import GRULayer


class GRU:

    def __init__(self, h_n):
        self.x_n = 256
        self.h_n = h_n
        self.layer = GRULayer(self.x_n, h_n)
        self.Wy = np.random.randn(self.x_n, h_n + 1) * 0.01
        self.mW, self.mWr, self.mWz, self.mWy = \
            [np.zeros_like(x) for x in [self.layer.W, self.layer.Wr, self.layer.Wz, self.Wy]]

    def train(self, s, it=100):
        seq_length = 90
        loss_report_n = 100

        quantized = compander.quantize(s)
        n = s.size
        p = 0
        layer = self.layer
        losses = np.zeros(loss_report_n)

        for i in range(it):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= n or i == 0:
                h_prev = np.zeros(layer.h_n)  # reset RNN memory
                p = 0  # go from start of data
                print('New epoch (it ', (i + 1), ')')

            inputs = np.zeros((seq_length, 256))
            for j in range(seq_length):
                inputs[j][quantized[p + j]] = 1

            h = layer.forward(inputs, initial_h=h_prev)
            dh = np.zeros_like(h)
            loss = 0
            dWy = np.zeros_like(self.Wy)

            for j in range(seq_length):
                y = np.dot(self.Wy, np.r_[h[j], np.ones(1)][:, np.newaxis]).flatten()
                if np.max(y) > 500:  # learning rate too high?
                    print('Warning: y value too large: ', np.max(y))
                    np.clip(y, None, 500, out=y)

                dy = np.exp(y) / np.sum(np.exp(y))  # also means the probabilities
                loss += -np.log(dy[quantized[p + 1 + j]])
                dy[quantized[p + 1 + j]] -= 1

                dWy += np.dot(dy[:, np.newaxis], np.r_[h[j], np.ones(1)][np.newaxis])
                dh[j] = np.dot(self.Wy.T, dy[:, np.newaxis]).flatten()[:-1]

            h_prev = np.copy(h[-1])
            layer.backward(dh)

            losses[i % loss_report_n] = loss
            if i % loss_report_n == (loss_report_n - 1):
                print(losses)
                print('iter:\t%d\tloss:\t%f' % (i + 1, np.mean(losses)))  # print progress

            alpha = 1e-1
            for param, dparam, mem in zip([layer.W, layer.Wr, layer.Wz, self.Wy],
                                          [layer.dW, layer.dWr, layer.dWz, dWy],
                                          [self.mW, self.mWr, self.mWz, self.mWy]):
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

            h = self.layer.forward(x[None], initial_h=h).flatten()
            y = np.dot(self.Wy, np.r_[h, 1][:, None]).flatten()

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
