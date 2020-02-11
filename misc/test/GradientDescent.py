#/*==========================================================================================*\
#**                        _           _ _   _     _  _         _                            **
#**                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
#**                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
#**                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
#**                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
#\*==========================================================================================*/


import math
import numpy as np;
import matplotlib.pyplot as plt

MAX_ITERATION = 1000
DEFAULT_EPS = 1e-4

# Function: f = x^2 + 5sin(x)
def func(x):
    return x**2 + 10*np.sin(x)

def grad(x):
    return 2*x + 10*np.cos(x)

def numerical_grad(x):
    x_eps = 1e-4
    y_eps = 1e-6
    ret = np.zeros_like(x)
    for i in range(len(x)):
        x_1 = x_2 = x.copy()
        x_1[i] += x_eps;
        x_2[i] -= x_eps;
        ret[i] = (func(x_1) - func(x_2)) / 2*x_eps;
    return ret

def has_converged(theta):
    return np.linalg.norm(grad(theta))/len(theta) < 1e-3

def check(x1, x2, color1, color2):
    _x = np.linspace(-5.5, 5.5, 100)
    _y = func(_x)
    plt.plot(_x, _y, 'b')

    y1 = func(x1)
    y2 = func(x2)
    plt.plot(x1, y1, color1)
    plt.plot(x2, y2, color2)
    plt.show()

#=========================================================
def standardGD(initial_point, learning_rate):
    x = [initial_point]
    for i in range(MAX_ITERATION):
        x_new = x[-1] - learning_rate * grad(x[-1])
        if abs(grad(x_new)) < 1e-3: break
        x.append(x_new)
    return (np.asarray(x), i)

def momentumGD(initial_point, learning_rate, gamma):
    x = [initial_point]
    v_old = np.zeros_like(initial_point)
    for i in range(MAX_ITERATION):
        v_new = learning_rate * grad(x[-1]) + gamma * v_old
        x_new = x[-1] - v_new

        # if has_converged(x): break
        if abs(grad(x_new)) < 1e-3: break
        x.append(x_new)
        v_old = v_new
    return (np.asarray(x), i)

def nesterovAG(initial_point, learning_rate, gamma):
    x = [initial_point]
    v_old = np.zeros_like(initial_point)
    for i in range(MAX_ITERATION):
        v_new = learning_rate * grad(x[-1] - gamma * v_old) + gamma * v_old
        x_new = x[-1] - v_new

        # if has_converged(x): break
        if abs(grad(x_new)) < 1e-3: break
        x.append(x_new)
        v_old = v_new
    return (np.asarray(x), i)

#=========================================================
(x1, it1) = standardGD(-5, .1)
(x2, it2) = standardGD(5, .1)
check(x1, x2, 'ro-', 'ro-')
#print('With Standard GD Algorithm, solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], func(x1[-1]), it1))
print('With Standard GD Algorithm, solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], func(x2[-1]), it2))


(x1, it1) = momentumGD(-5, .1, .9)
(x2, it2) = momentumGD(5, .1, .9)
check(x1, x2, 'ro-', 'ro-')
#print('With Momentum GD Algorithm, solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], func(x1[-1]), it1))
print('With Momentum GD Algorithm, solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], func(x2[-1]), it2))


(x1, it1) = nesterovAG(-5, .1, .9)
(x2, it2) = nesterovAG(5, .1, .9)
check(x1, x2, 'ro-', 'ro-')
#print('With Nesterov AG Algorithm, solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], func(x1[-1]), it1))
print('With Nesterov AG Algorithm, solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], func(x2[-1]), it2))

#plt.show()