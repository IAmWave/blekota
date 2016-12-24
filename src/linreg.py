import numpy as np
from sklearn import linear_model

clf = linear_model.Ridge(alpha=0) # normalization
#linear_model.SGDRegressor()
window = 200

def soundToData(s):
    n = s.size
    X = np.zeros((n, window))
    y = np.concatenate((np.zeros(window), s))
    for i in range(0, n):
        X[i] = y[i:(i+window)]
    return X


def train(s):
    X = soundToData(s)
    clf.fit(X, s)

def predict(s):
    X = soundToData(s)
    return clf.predict(X)


memory = np.zeros(window)


def init():
    memory = np.zeros(window)

def generate():
    global memory
    pred = clf.predict(memory.reshape(1, -1))
    memory = np.append(memory[1:], pred)
    return pred[0]


#y_predicted = clf.predict(X_to_predict)