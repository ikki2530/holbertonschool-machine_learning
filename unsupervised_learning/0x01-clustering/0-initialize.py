#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
    - X is a numpy.ndarray of shape (n, d) containing the
    dataset that will be used for K-means clustering.
        * n is the number of data points
        * is the number of dimensions for each data point
    - k is a positive integer containing the number of clusters.
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    centroids = np.random.uniform(mins, maxs, size=(k, X.shape[1]))
    return centroids
