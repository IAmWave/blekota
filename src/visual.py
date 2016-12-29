import numpy as np
import matplotlib.pyplot as plt
import const

def compare(y, y2, fs=const.DEFAULT_FS):
    n = y.size
    x = np.linspace(0, n / fs, n) # np.arange(n)
    
    plt.figure()
    plt.subplot(211)
    plt.title('Waveform comparison')
    plt.plot(x, y2, 'r', linewidth=4.0)
    plt.plot(x, y, 'k--', linewidth=2.0)
    plt.xlim(0, 0.05)
    plt.xlabel('Time')
    plt.ylabel('Sound wave')

    running_average_n = 100
    def running_average(arr):
        return np.convolve(arr, np.ones((running_average_n,))/running_average_n, mode='valid')

    err = running_average((y - y2) ** 2)
    err_bad = running_average(y ** 2)    # error when predicting 0

    plt.subplot(212)
    plt.title('Square error over time')
    plt.plot(err[::running_average_n], 'r')
    plt.plot(err_bad[::running_average_n], 'k')
    plt.axhline(0, color='black')
    plt.xlabel('Time')
    plt.ylabel('Square error')

    plt.show()

def show(y, fs=const.DEFAULT_FS):
    n = y.size
    x = np.linspace(0, n / fs, n) # np.arange(n)
    
    fig = plt.figure()
    constrainXPanZoomBehavior(fig)
    plt.title('Waveform')
    plt.axhline(0, color='black')
    plt.plot(x, y, 'r', linewidth=2.0)
    plt.xlim(0, 0.05)
    plt.xlabel('Time')
    plt.ylabel('Sound wave')
    plt.show()
