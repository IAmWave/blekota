import numpy as np
from sklearn.linear_model import LinearRegression

import audio_io
import linreg
import rnn
import gru
import visual

y, fs = audio_io.load_file('data/shapes/warble2.wav')
time = 10
y = y[:800000 * 5]  # 100 seconds at most
clf = gru.GRU(50, layer_n=3)
print("Training...")

clf.train(y, it=2000)
print("Done")

y = y[0:int(fs * time)]
n = y.size

# rnn.hprev = np.zeros((hidden_size,1)) # reset RNN memory
#y2 = (rnn.sample(np.zeros((rnn.hidden_size,1)), 128, y.size) - 128) / 128
y2 = clf.sample(n, hint=y[:1])
# y2 = clf.sample(n, hint=y[:1])

#np.concatenate((np.zeros(2000), y2))
#audio_io.play(y2, blocking=False)
#visual.compare(y, y2, fs=fs)
