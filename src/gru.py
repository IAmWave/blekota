import numpy as np
import compander
from gru_layer import GRULayer
from param import Param


class GRU:

    def __init__(self, h_n, layer_n=1, seq_length=160, batches=60, alpha=2e-3, loss_report_n=50):
        self.h_n = h_n
        self.layer_n = layer_n
        self.seq_length = seq_length
        self.batches = batches
        self.alpha = alpha
        self.loss_report_n = loss_report_n

        self.epochs = 0
        self.iterations = 0

        self.layers = [GRULayer(256, h_n, name='0')]  # first layer has different dimensions
        for i in range(self.layer_n - 1):
            self.layers.append(GRULayer(h_n, h_n, name=str(i + 1)))

        self.Wy = Param(np.random.randn(256, h_n + 1) * 0.01, name='Wy')

        self.params = [self.Wy]
        for i in range(self.layer_n):
            self.params += self.layers[i].getParams()

    def train(self, sound, it=100):
        print('Training for', it, 'iterations')
        x = np.resize(compander.quantize(sound), (self.batches, sound.size // self.batches)).T
        n = x.shape[0]
        l_n = self.layer_n
        p = 0
        losses = np.zeros(self.loss_report_n)

        layers = self.layers
        h_prev = {}

        for i in range(it):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.seq_length + 1 >= n or i == 0:
                if i != 0:
                    self.epochs += 1
                # reset RNN memory
                for l in range(l_n):
                    h_prev[l] = np.zeros((self.h_n, self.batches))

                p = 0  # go from start of data
                # print('New epoch (it ', (i + 1), ')')

            inputs = np.zeros((self.seq_length, 256, self.batches))
            for t in range(self.seq_length):
                for k in range(self.batches):
                    inputs[t, x[p + t, k], k] = 1

            hs, dhs = {}, {}

            for l in range(l_n):
                hs[l] = layers[l].forward(inputs if (l == 0) else hs[l - 1], initial_h=h_prev[l])
                h_prev[l] = np.copy(layers[l].h[-1])

            dhs[l_n - 1] = np.zeros_like(hs[l_n - 1])
            loss = 0

            for t in range(self.seq_length):
                y = np.dot(self.Wy.a, np.r_[hs[l_n - 1][t], np.ones((1, self.batches))])

                if np.max(y) > 500:  # learning rate too high?
                    print('Warning: y value too large: ', np.max(y))
                    np.clip(y, None, 500, out=y)

                dy = np.exp(y) / np.sum(np.exp(y), axis=0)  # also means the probabilities
                for k in range(self.batches):
                    loss += -np.log(dy[x[p + 1 + t, k], k])
                    dy[x[p + 1 + t, k], k] -= 1  # different derivative at the correct answer

                self.Wy.d += np.dot(dy, np.r_[hs[l_n - 1][t], np.ones((1, self.batches))].T)
                dhs[l_n - 1][t] = np.dot(self.Wy.a.T, dy)[:-1, :]

            loss /= self.batches  # makes comparing performance easier

            for l in reversed(range(self.layer_n)):
                dhprev = layers[l].backward(dhs[l])
                if l > 0:
                    dhs[l - 1] = np.copy(dhprev)

            losses[i % self.loss_report_n] = loss
            if i % self.loss_report_n == (self.loss_report_n - 1):
                if i < (self.loss_report_n) * 5:  # only print full losses the first few times
                    print(losses)
                print('iter:\t%d\tloss:\t%f' % (i + 1, np.mean(losses)))  # print progress

            for param in self.params:
                param.step(self.alpha)

            p += self.seq_length  # move data pointer
            self.iterations += 1

    def sample(self, n, hint=np.zeros(1), temperature=1):
        layers = self.layers
        l_n = self.layer_n
        h = {}
        for l in range(l_n):
            h[l] = np.zeros((self.h_n, 1))

        x = np.zeros(256)
        x[compander.quantize(hint[0])] = 0
        res = np.zeros(n)
        p_save_n = min(n, 8000 * 5)
        p = np.zeros((p_save_n, 256))

        for t in range(n):
            for l in range(l_n):
                h[l] = layers[l].forward(x[None, :, None] if (l == 0) else h[l - 1][None, :], initial_h=h[l])[0]

            y = np.dot(self.Wy.a, np.r_[h[l_n - 1], np.ones((1, 1))])
            y /= temperature
            p_cur = (np.exp(y) / np.sum(np.exp(y))).ravel()
            chosen = np.random.choice(range(256), p=p_cur)
            if t < p_save_n:
                p[t] = p_cur

            x = np.zeros(256)
            # use hint output if available
            x[compander.quantize(hint[t]) if (t < hint.shape[0]) else chosen] = 1

            res[t] = chosen

        self.p = p
        return compander.unquantize(res)

    def __repr__(self):
        res = "GRU"
        for name, value in zip(["Layers", "Hidden", "Batches", "Alpha", "Seq length", "Iterations", "Epochs"],
                               [self.layer_n, self.h_n, self.batches, self.alpha, self.seq_length, self.iterations, self.epochs]):
            res += "\n{:12}{}".format(name, value)

        return res

    def __getstate__(self):  # for pickling
        self.p = np.zeros(1)  # huge space waste; unnecessary
        return self.__dict__
