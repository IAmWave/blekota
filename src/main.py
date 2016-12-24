import numpy as np
from sklearn.linear_model import LinearRegression

import audio_io
import linreg
import visual

y, fs = audio_io.load_file('data/shapes/warble_1000-2000Hz_-6dBFS_3s.wav')
y = y[0:8000]
n = y.size

print("Training...")
linreg.train(y)
print("Done")

y2 = np.zeros(n)
y2[0:linreg.window] = y[0:linreg.window]


linreg.init()
linreg.memory = y[0:linreg.window]
for i in range(n - linreg.window):
    cur = linreg.generate()
    y2[i+linreg.window] = cur

#y2 = linreg.predict(y)

audio_io.play(y2, blocking=False)
audio_io.play(y, blocking=True)
visual.compare(y, y2, fs=fs)
