#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    - X is a numpy.ndarray of shape (nx, m) containing
    the input data for the network, nx is the number of input features and
    m is the number of data points.
    - weights is a dictionary of the weights and biases of the neural network.
    - L the number of layers in the network.
    - keep_prob is the probability that a node will be kept.
    - All layers except the last should use the tanh activation function.
    - The last layer should use the softmax activation function.
    Returns: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer (see example for format).
    """
    cache = {}
    cache["A" + str(0)] = X
    for i in range(0, L):
        z = weights["W" + str(i + 1)] @ cache["A" + str(i)] + weights[
                    "b" + str(i + 1)]
        if i == L - 1:
            # softmax
            e_x = np.exp(z - np.max(z))
            cache["A" + str(i + 1)] = (e_x / e_x.sum(axis=0))
        else:
            # tanh
            a = (np.tanh(z)) / keep_prob
            d = np.random.rand(a.shape[0], a.shape[1])
            d = np.where(d < keep_prob, 1, 0)  # mask
            cache["D" + str(i + 1)] = d
            cache["A" + str(i + 1)] = a * d
    return cache
