#!/usr/bin/env python3

import numpy as np

Neuron = __import__('1-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T


np.random.seed(0)
neuron = Neuron(X.shape[0])
print(X)
#print(dir(neuron))
#print(neuron.W)
print(neuron.b)
print(neuron.A)
print("-------------------------------")
neuron.A = 10
print(neuron.A)

