#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural
network using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    - Normalizes an unactivated output of a neural
    network using batch normalization.
    - Z is a numpy.ndarray of shape (m, n) that should be normalized,
    m is the number of data points and n is the number of features in Z.
    - gamma is a numpy.ndarray of shape (1, n) containing the scales
    used for batch normalization.
    - beta is a numpy.ndarray of shape (1, n) containing the
    offsets used for batch normalization.
    - epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)  # 1 x n
    var = np.var(Z, axis=0)  # 1 x n
    znorm = (Z - mean) / np.sqrt(var + epsilon)
    zv = gamma * znorm + beta
    return zv
