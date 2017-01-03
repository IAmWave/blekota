import numpy as np
from sklearn.linear_model import LinearRegression

import audio_io
import linreg
import rnn
import gru
import visual

y, fs = audio_io.load_file('data/shapes/warble_1000-2000Hz_-6dBFS_3s.wav')
time = 3
#y = np.zeros_like(y) + 0.4

clf = gru.GRU(100)  # rnn.RNN()  # gru.GRU(256)
print("Training...")
# rnn.train(y)
clf.train(y, it=5000)
print("Done")

y = y[0:int(fs * time)]
n = y.size

# rnn.hprev = np.zeros((hidden_size,1)) # reset RNN memory
#y2 = (rnn.sample(np.zeros((rnn.hidden_size,1)), 128, y.size) - 128) / 128
y2 = clf.sample(n, hint=y[0:1])

#np.concatenate((np.zeros(2000), y2))
audio_io.play(y2, blocking=False)
#visual.compare(y, y2, fs=fs)
visual.show(y2, fs=fs)
