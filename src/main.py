import numpy as np
from sklearn.linear_model import LinearRegression

import audio_io
import linreg
import rnn
import gru
import visual

y, fs = audio_io.load_file('data/shapes/sin_440Hz_8kHz_3s.wav')
time = 0.1

clf = gru.GRU(256)
print("Training...")
# rnn.train(y)
clf.train(y, it=200)
print("Done")

y = y[0:int(fs * time)]
n = y.size

# rnn.hprev = np.zeros((hidden_size,1)) # reset RNN memory
#y2 = (rnn.sample(np.zeros((rnn.hidden_size,1)), 128, y.size) - 128) / 128
y2 = clf.sample(n)

#audio_io.play(np.concatenate((np.zeros(2000), y2)), blocking=False)
import compander
y = compander.unquantize(compander.quantize(y))
visual.compare(y, y2, fs=fs)
#visual.show(y2, fs=fs)
