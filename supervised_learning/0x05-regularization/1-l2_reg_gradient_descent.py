#!/usr/bin/env python3
"""
updates the weights and biases of a neural network using gradient
descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    - Updates the weights and biases of a neural
    network using gradient descent with L2 regularization.
    - Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data, classes
    is the number of classes and m is the number of data points.
    - weights is a dictionary of the weights and biases of the neural network.
    - cache is a dictionary of the outputs of each layer of the neural network.
    - alpha is the learning rate.
    - lambtha is the L2 regularization parameter.
    - L is the number of layers of the network.
    - The neural network uses tanh activations on each layer
    except the last, which uses a softmax activation.
    - The weights and biases of the network should be updated in place.
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        A = cache["A" + str(i)]
        if i == L:
            dz = A - Y
        else:
            g = 1 - (A ** 2)
            dz = (weights_copy["W" + str(i + 1)].T @ dz) * g
        dw = (dz @ cache["A" + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        # dw is the backpropagation
        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * ((
                lambtha / m) * weights["W" + str(i)] + dw))
        weights["b" + str(i)] = weights["b" + str(i)] - (alpha * db)
