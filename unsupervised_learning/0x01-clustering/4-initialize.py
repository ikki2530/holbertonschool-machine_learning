#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if type(k) != int or k <= 0:
        return None, None, None

    n, d = X.shape
    pi = np.tile(1/k, (k,))
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
