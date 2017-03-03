import numpy as np
import pickle

import audio_io
import linreg
import rnn
import gru
import visual


def save(name):
    pickle.dump(clf, open(name, 'wb'))


def heatmap(start=0, length=1000):
    visual.heatmap(clf.p[start:(start + length)])

show = visual.show
play = audio_io.play

file = 'data/voice/vali/vali_16k.wav'

y, fs = audio_io.load_file(file)
time = 1
clf = gru.GRU(256, layer_n=5)
clf.file = file
#clf = pickle.load(open('pi4l_2.pkl', 'rb'))
print(clf)
print('Data file:', file)
print("Training...")

clf.train(y, it=10000)
print("Done")

#y2 = clf.sample((fs * time), hint=y[:1])
