#!/usr/bin/env python3
"""
updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    - Updates the weights of a neural network with Dropout regularization
    using gradient descent.
    - Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data, classes is the number
    of classes and m is the number of data points.
    - weights is a dictionary of the weights and biases of the neural network.
    - cache is a dictionary of the outputs and dropout masks of each layer
    of the neural network.
    - alpha is the learning rate.
    - keep_prob is the probability that a node will be kept.
    - L is the number of layers of the network.
    - The weights of the network should be updated in place
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        A = cache["A" + str(i)]
        if i == L:
            dz = A - Y
        else:
            # tanh
            g = 1 - (A ** 2)
            dz = ((weights_copy["W" + str(i + 1)].T @ dz) * g) / keep_prob
        dw = (dz @ cache["A" + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        # dw is the backpropagation
        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * (dw))
        weights["b" + str(i)] = weights["b" + str(i)] - (alpha * db)
