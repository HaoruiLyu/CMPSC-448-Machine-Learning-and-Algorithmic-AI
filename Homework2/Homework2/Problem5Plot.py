import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import *


def bgd_plot(data, y, w, eta, delta, lam, num_iter):
    w, f = bgd_l2(data, y, w, eta, delta, lam, num_iter)
    plt.plot(f)
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function history')
    plt.show()


def sgd_plot(data, y, w, eta, delta, lam, num_iter):
    w, f = sgd_l2(data, y, w, eta, delta, lam, num_iter)
    plt.plot(f)
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function history')
    plt.show()


if __name__ == '__main__':
    data = np.load('data.npy')
    x = data[:, 0]
    y = data[:, 1]

    bias = np.ones((100,), dtype=int)
    data2 = np.column_stack((x, bias))

    w = np.array(([0, 0]), dtype=float)

    bgd_plot(data2, y, w, 0.05, 0.1, 0.001, 50)
    bgd_plot(data2, y, w, 0.1, 0.01, 0.001, 50)
    bgd_plot(data2, y, w, 0.1, 0, 0.001, 100)
    bgd_plot(data2, y, w, 0.1, 0, 0, 100)

    print(1)
    sgd_plot(data2, y, w, 1, 0.1, 0.5, 800)
    sgd_plot(data2, y, w, 1, 0.01, 0.1, 800)
    sgd_plot(data2, y, w, 1, 0, 0, 40)
    sgd_plot(data2, y, w, 1, 0, 0, 800)
