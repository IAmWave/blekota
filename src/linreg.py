import numpy as np
from sklearn import linear_model

class LinReg:

    def __init__(self, window=50):
        self.clf = linear_model.Ridge(alpha=0) # normalization
        #linear_model.SGDRegressor()
        self.window = window
        self.memory = np.zeros(window)
    
    def soundToData(self, s):
        n = s.size
        X = np.zeros((n, self.window))
        y = np.concatenate((np.zeros(self.window), s))
        for i in range(0, n):
            X[i] = y[i:(i+self.window)]
        return X


    def train(self, s, it=0):
        X = self.soundToData(s)
        self.clf.fit(X, s)

    def predict(self, s):
        X = self.soundToData(s)
        return self.clf.predict(X)

    def step(self):
        pred = self.clf.predict(self.memory.reshape(1, -1))
        self.memory = np.append(self.memory[1:], pred)
        return pred[0]

    def sample(self, n, hint=np.zeros(1)):
        y = np.zeros(n)
        y[0:self.window] = hint[0:self.window]
        self.memory = y[0:self.window]
        for i in range(n - self.window):
            cur = self.step()
            y[i + self.window] = cur
        return y

    #y_predicted = clf.predict(X_to_predict)