"""
Based on "Minimal character-level Vanilla RNN model" self.by Andrej Karpathy (@karpathy)
BSD License
https://gist.github.com/karpathy/d4dee566867f8291f086
"""
import numpy as np
import math
import compander

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 90  # number of steps to unroll the RNN for
learning_rate = 1e-1
vocab_size = 256


class RNN:

    def __init__(self):
        # model parameters
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((vocab_size, 1))  # output bias

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ps = {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # hidden state
            y = np.dot(self.Why, hs[t]) + self.by  # unnormalized log probabilities for next chars
            ps[t] = np.exp(y) / np.sum(np.exp(y))  # probabilities for next chars
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        #print('mean ', np.mean(dWhh ** 2))
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def sample(self, n, hint=np.zeros(1), h=np.zeros((hidden_size, 1))):
        """ 
        sample a sequence of integers from the model 
        h is memory state, seed_ix is seed letter for first time step
        """
        #x = (np.arange(vocab_size))
        x = np.zeros((vocab_size, 1))
        x[compander.quantize(hint[0])] = 1
        p = np.zeros((n, 256))

        ixes = np.zeros(n)
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p[t] = (np.exp(y) / np.sum(np.exp(y))).flatten()
            ix = np.random.choice(range(vocab_size), p=p[t].ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes[t] = ix

        self.p = p
        return compander.unquantize(ixes)

    def train(self, s, it=1000):
        loss_report_n = 100
        data = compander.quantize(s)

        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        losses = np.zeros(loss_report_n)

        while n < it:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= len(data) or n == 0:
                hprev = np.zeros((hidden_size, 1))  # reset RNN memory
                p = 0  # go from start of data
            inputs = data[p:p + seq_length]
            targets = data[p + 1:p + seq_length + 1]
            """
            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.sample(200, hint=inputs, h=hprev)
                print(sample_ix)
                #txt = ' '.join(ix + ' ' for ix in sample_ix)
                #print('----\n %s \n----' % (txt, ))
            """
            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)

            losses[n % loss_report_n] = loss
            if n % loss_report_n == (loss_report_n - 1):
                print(losses)
                print('iter:\t%d\tloss:\t%f' % (n + 1, np.mean(losses)))  # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh,  self.Whh,  self.Why,  self.bh,  self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += seq_length  # move data pointer
            n += 1  # iteration counter
