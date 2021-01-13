#!/usr/bin/env python3
"""Normalize"""
import numpy as np


def shuffle_data(X, Y):
    """
    - Shuffles the data points in two matrices the same way.
    - X is the first numpy.ndarray of shape (m, nx) to shuffle,
    m is the number of data points and nx is the number of features in X.
    - Y is the second numpy.ndarray of shape (m, ny) to shuffle,
    m is the same number of data points as in X and ny is the number
    of features in Y.
    - Returns: the shuffled X and Y matrices.
    """
    shuff = np.random.permutation(X.shape[0])
    X_s = X[shuff]
    Y_s = Y[shuff]
    return X_s, Y_s
