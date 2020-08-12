import test
import os
import numpy as np
from cvxopt import solvers, matrix, spdiag, log
import copy
import random

import itertools

import matplotlib.pyplot as plt

from itertools import combinations, permutations

# from array import  array

from parameter_setting import args

import multiprocessing as mp

import seaborn as sns



def plot_test():
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    print("t", t)
    print("t ** 2", t**2)
    print("t ** 3", t ** 3)


    # red dashes, blue squares and green triangles
    plt.plot(t, t)

    # plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()
    plt.close()
    plt.savefig('plot_file/jake_test_1.eps', format='eps')

    return

def plot_test_2():
    '''
    Demonstration of wind barb plots
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    U, V = 12 * X, 12 * Y

    data = [(-1.5, .5, -6, -6),
            (1, -1, -46, 46),
            (-3, -1, 11, -11),
            (1, 1.5, 80, 80),
            (0.5, 0.25, 25, 15),
            (-1.5, -0.5, -5, 40)]

    data = np.array(data, dtype=[('x', np.float32), ('y', np.float32),
                                 ('u', np.float32), ('v', np.float32)])

    # Default parameters, uniform grid
    ax = plt.subplot(2, 2, 1)
    ax.barbs(X, Y, U, V)

    # Arbitrary set of vectors, make them longer and change the pivot point
    # (point around which they're rotated) to be the middle
    ax = plt.subplot(2, 2, 2)
    ax.barbs(data['x'], data['y'], data['u'], data['v'], length=8, pivot='middle')

    # Showing colormapping with uniform grid.  Fill the circle for an empty barb,
    # don't round the values, and change some of the size parameters
    ax = plt.subplot(2, 2, 3)
    ax.barbs(X, Y, U, V, np.sqrt(U * U + V * V), fill_empty=True, rounding=False,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3))

    # Change colors as well as the increments for parts of the barbs
    ax = plt.subplot(2, 2, 4)
    ax.barbs(data['x'], data['y'], data['u'], data['v'], flagcolor='r',
             barbcolor=['b', 'g'], barb_increments=dict(half=10, full=20, flag=100),
             flip_barb=True)

    # Masked arrays are also supported
    masked_u = np.ma.masked_array(data['u'])
    masked_u[4] = 1000  # Bad value that should not be plotted when masked
    masked_u[4] = np.ma.masked

    # Identical plot to panel 2 in the first figure, but with the point at
    # (0.5, 0.25) missing (masked)
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.barbs(data['x'], data['y'], masked_u, data['v'], length=8, pivot='middle')

    plt.show()

def plot_test_3():
    fig1 = plt.figure()
    fig = plt.figure()
    x = np.arange(10)
    # y = 2.5 * np.sin(x / 20 * np.pi)
    y = x
    yerr = np.linspace(0.05, 0.2, 10)

    plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')
    # plt.errorbar(x, y + 3, label='both limits (default)')
    plt.show()

def print_aa():
    print ("hahhahahaha in test2.py")



def test_heatmap():
    np.random.seed(0)
    len = 10
    x = np.random.randn(len, len)
    for i in range(len):
        for j in range(len):
            x[i][j] = i * 10 + j * 10
    x = x.astype(int)
    sns.set()
    f, ax1 = plt.subplots()
    sns.heatmap(x, annot=True, ax=ax1, fmt= 'd', cmap='YlGnBu')
    plt.show()



if __name__ == '__main__':

    # aa = 'aaaa'
    # print('len:', len(aa))
    # cc = aa.split('a')
    # print ("cc:", cc)
    # print('cc_len:', len(cc))
    # for tmp in cc:
    #     if tmp == '':
    #         print ("yes", len(tmp))
    #
    # print("cpu_count:", mp.cpu_count())

    # test_heatmap()

    aa = np.array([-1, -2, 5])
    bb = np.array([2,88, 99])

    cc = []
    cc.append(aa)
    cc.append(bb)

    dd = np.array(cc)
    print ('dd:', dd, 'len:', len(dd))






