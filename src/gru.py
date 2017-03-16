import numpy as np
import pickle

import compander
from gru_layer import GRULayer
from param import Param
import audio_io
import const


class GRU:
    """A complete GRU model, comprised of one or more layers.

    Very loosely based on Andrej Karpathy's code: https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    loss_report_n = 50

    def __init__(self, h_n, layer_n=1, seq_length=80, batches=80,
                 alpha=2e-3, loss_report_n=50, name='model', fs=const.DEFAULT_FS, file=''):
        """Initialize a GRU (with reasonable defaults).

        Args:
            h_n (int): The hidden vector size for each layer.
            layer_n (int): The number of layers.
            seq_length (int): How many steps to make before updating the model parameters.
            batches (int): How many parts to split the sound into and process in parallel.
                More batches should be more efficient and have a more stable loss function,
                but take more memory.
            alpha (float): The learning rate. Should not be much higher than the default;
                a higher alpha can lead to stagnation.
            loss_report_n (int): Once per how many iterations should the model report the loss.
            name (str): The name of the model - used as file prefix when saving the model via checkpoint()
            fs (int): Sampling rate of the sound. Relevant when sampling.
            file (str): The sound file on which the model is trained. The GRU class does not use
                this directly, but it is useful for knowing what we trained on.
        """
        self.h_n = h_n
        self.layer_n = layer_n
        self.seq_length = seq_length
        self.batches = batches
        self.alpha = alpha
        self.loss_report_n = loss_report_n
        self.name = name
        self.fs = fs
        self.file = file

        self.epochs = 0
        self.iterations = 0
        pos = 0

        self.layers = [GRULayer(const.SAMPLE_VALUES, h_n, name='0')]  # first layer has different dimensions
        for i in range(self.layer_n - 1):
            self.layers.append(GRULayer(h_n, h_n, name=str(i + 1)))

        # A learnable matrix applied as the final transformation of the last layer's h
        self.Wy = Param(np.random.randn(const.SAMPLE_VALUES, h_n + 1) * 0.01, name='Wy')

        self.params = [self.Wy]
        for i in range(self.layer_n):
            self.params += self.layers[i].getParams()

    def train(self, it):
        """Train the model for a set number of iterations.

        Args:
            it (int): The number of iterations for which the model should be trained.
        """
        print('Training for', it, 'iterations')

        sound, self.fs = audio_io.load_file(self.file)
        # quantize and split into batches
        x = np.resize(compander.quantize(sound), (self.batches, sound.size // self.batches)).T
        n = x.shape[0]
        l_n = self.layer_n
        losses = np.zeros(self.loss_report_n)

        layers = self.layers
        h_prev = {}
        pos = 0

        for i in range(it):

            if pos + self.seq_length + 1 >= n or i == 0:  # go from the start
                if i != 0:
                    self.epochs += 1
                # reset GRU memory
                for l in range(l_n):
                    h_prev[l] = np.zeros((self.h_n, self.batches))

                pos = 0

            # one-hot encode part of input (do not encode the whole input for memory reasons)
            inputs = np.zeros((self.seq_length, const.SAMPLE_VALUES, self.batches))
            for t in range(self.seq_length):
                for k in range(self.batches):
                    inputs[t, x[pos + t, k], k] = 1

            hs, dhs = {}, {}  # hidden state and its gradient for each layer

            # perform forward passes for each layer
            for l in range(l_n):
                hs[l] = layers[l].forward(inputs if (l == 0) else hs[l - 1], initial_h=h_prev[l])
                h_prev[l] = np.copy(layers[l].h[-1])

            dhs[l_n - 1] = np.zeros_like(hs[l_n - 1])
            loss = 0

            # compute gradient of the output and then the last layer's h (dhs[l_n-1])
            for t in range(self.seq_length):
                # multiply Wy by the last layer's h to get the log weights
                y = np.dot(self.Wy.a, np.r_[hs[l_n - 1][t], np.ones((1, self.batches))])

                if np.max(y) > 500:  # y should not be this large. Perhaps lower alpha?
                    print('Warning: y value too large: ', np.max(y))
                    np.clip(y, None, 500, out=y)

                # dy computation
                dy = np.exp(y) / np.sum(np.exp(y), axis=0)  # apply softmax to get probabilities
                for k in range(self.batches):
                    loss += -np.log(dy[x[pos + 1 + t, k], k])
                    dy[x[pos + 1 + t, k], k] -= 1  # different derivative at the correct answer
                # gradients of Wy and hs[l_n - 1] (h of last layer)
                self.Wy.d += np.dot(dy, np.r_[hs[l_n - 1][t], np.ones((1, self.batches))].T)
                dhs[l_n - 1][t] = np.dot(self.Wy.a.T, dy)[:-1, :]

            # normalize the loss - makes comparing performance easier
            loss /= self.batches * self.seq_length

            # gradients of previous layers
            for l in reversed(range(self.layer_n)):
                dhprev = layers[l].backward(dhs[l])
                if l > 0:
                    dhs[l - 1] = np.copy(dhprev)

            # print the losses
            losses[i % self.loss_report_n] = loss
            if i % self.loss_report_n == (self.loss_report_n - 1):
                if i < (self.loss_report_n) * 5:  # only print full losses the first few times
                    print(losses)
                print('Iteration:\t%d\tLoss:\t%f' % (self.iterations + 1, np.mean(losses)))

            # update all parameters
            for param in self.params:
                param.step(self.alpha)

            pos += self.seq_length  # move data pointer
            self.iterations += 1

            # save progress once in a while
            if self.iterations % 1000 == 0:
                self.checkpoint()

    def checkpoint(self, sampleLength=50000):
        """Save the model to a file and optionally sample it as well.

        Args:
            sampleLength (int, optional): How long should be the sampled sound be.
                No sample is taken when sampleLength<=0.
        """

        pickle.dump(self, open(self.name + '_' + str(self.iterations) + '.pkl', 'wb'))
        if sampleLength > 0:
            y2 = self.sample(sampleLength)
            audio_io.save_file(self.name + '_' + str(self.iterations) + '.wav', y2, self.fs)

    def sample(self, n, hint=np.zeros(1), temperature=1):
        """Sample from the model.

        Args:
            n (int): How many samples to take (length of the generated sound).
            hint (array, optional): A source sound to serve as the beginning of the input.
                When hint is unavailable, the model's output at time t is used as the input of time t+1.
            temperature (float, optional): Weights are raised to the power (1/temperature).
                A higher temperature means more similar probabilities and thus more varied output.
                Should be greater than 0.

        Returns:
            An array of length n; the sampled sound.
        """
        layers = self.layers
        l_n = self.layer_n
        h = {}
        for l in range(l_n):
            h[l] = np.zeros((self.h_n, 1))

        x = np.zeros(const.SAMPLE_VALUES)
        x[compander.quantize(hint[0])] = 0
        res = np.zeros(n)
        p_save_n = min(n, 25000)  # only save a small part (memory reasons)
        p = np.zeros((p_save_n, const.SAMPLE_VALUES))

        for t in range(n):
            # forward passes of each layer
            for l in range(l_n):
                cur_x = x[None, :, None] if (l == 0) else h[l - 1][None, :]
                h[l] = layers[l].forward(cur_x, initial_h=h[l])[0]

            y = np.dot(self.Wy.a, np.r_[h[l_n - 1], np.ones((1, 1))])
            y /= temperature  # higher temperature = more equal probabilities
            p_cur = (np.exp(y) / np.sum(np.exp(y))).ravel()
            # pick one value based on the computed probability distribution
            chosen = np.random.choice(range(const.SAMPLE_VALUES), p=p_cur)
            if t < p_save_n:
                p[t] = p_cur

            # prepare next input - use hint output if available
            x = np.zeros(const.SAMPLE_VALUES)
            x[compander.quantize(hint[t]) if (t < hint.shape[0]) else chosen] = 1

            res[t] = chosen

        self.p = p
        # unquantize - put the sound back into playable form
        return compander.unquantize(res)

    def __repr__(self):
        """Pretty-print (string representation)"""
        res = "GRU"
        for name, value in zip(["Name", "File", "Layers", "Hidden", "Batches", "Alpha", "Seq length", "Iterations"],
                               [self.name, self.file, self.layer_n, self.h_n, self.batches, self.alpha, self.seq_length, self.iterations]):
            res += "\n{:12}{}".format(name, value)

        return res

    def __getstate__(self):
        """Return a representation of the object for pickling - omit huge p attribute"""
        self.p = np.zeros(1)  # huge space waste; unnecessary
        return self.__dict__
