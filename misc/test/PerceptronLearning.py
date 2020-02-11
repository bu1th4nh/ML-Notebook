import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)
import time

gamma = 1
# Data generation
means = [[1, 2], [3, 2]]
cov = [[.3, .2], [.2, .3]]
N = 90  # numbers of data points
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

print("dataGeneration done.")

# Perceptron Algorithm
def h(w, x): return np.sign(np.dot(w.T, x))
def has_converged(X, y, w): return np.array_equal(h(w, X), y)
def perceptronRegularGD(X, y, w_init):
    start_time = time.time()

    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_point = []
    v_old = np.zeros_like(w_init)

    while 1:
        mix_id = np.random.permutation(N)
        for i in range(N):
            x_i = X[:, mix_id[i]].reshape(d, 1)
            y_i = y[0, mix_id[i]]
            if(h(w[-1], x_i)[0] != y_i):
                mis_point.append(mix_id[i])
                # Regular GD
                w_new = w[-1] + y_i * x_i

                # Nesterov Accelerated GD
                # v_new = v_old * gamma - y_i * (x_i - v_old * gamma)
                # w_new = w[-1] - v_new
                w.append(w_new)
        if has_converged(X, y, w[-1]): break

    end_time = time.time()
    print("Perceptron-Std. algorithm completed w/ running time: %.4f (s)" % (end_time - start_time))
    return (w, mis_point)

def perceptronNesterovGD(X, y, w_init):
    start_time = time.time()

    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_point = []
    v_old = np.zeros_like(w_init)

    while 1:
        mix_id = np.random.permutation(N)
        for i in range(N):
            x_i = X[:, mix_id[i]].reshape(d, 1)
            y_i = y[0, mix_id[i]]
            if(h(w[-1], x_i)[0] != y_i):
                mis_point.append(mix_id[i])
                # Regular GD
                # w_new = w[-1] + y_i * x_i

                # Nesterov Accelerated GD
                v_new = v_old * gamma - y_i * (x_i - v_old * gamma)
                w_new = w[-1] - v_new
                w.append(w_new)
        if has_converged(X, y, w[-1]): break

    end_time = time.time()
    print("Perceptron-NAG. algorithm completed w/ running time: %.4f (s)" % (end_time - start_time))
    return (w, mis_point)


# print(w)
# print(len(w))

def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1 * x11 + w0) / w2, -(w1 * x12 + w0) / w2], 'k')
    else:
        x10 = -w0 / w1
        return plt.plot([x10, x10], [-100, 100], 'k')
## Visualization
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
def viz_alg_1d_2(w):
    it = len(w)
    fig, ax = plt.subplots(figsize=(5, 5))

    it = len(w)
    fig, ax = plt.subplots(figsize=(5, 5))

    ani = plt.cla()
    ani = plt.plot(X0[0, :], X0[1, :], 'b^', markersize=8, alpha=.8)
    ani = plt.plot(X1[0, :], X1[1, :], 'ro', markersize=8, alpha=.8)
    ani = plt.axis([-1, 5.5, -1, 4.5])
    ani = draw_line(w[-1])

    plt.show()

d = X.shape[0]
w_init = np.random.randn(d, 1)

(w, m) = perceptronRegularGD(X, y, w_init)
print(m)
viz_alg_1d_2(w)

(w, m) = perceptronNesterovGD(X, y, w_init)
print(m)
viz_alg_1d_2(w)

