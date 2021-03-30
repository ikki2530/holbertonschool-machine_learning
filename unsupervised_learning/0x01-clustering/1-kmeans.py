#!/usr/bin/env python3
"""
Performs K-means on a dataset.
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


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    - X is a numpy.ndarray of shape (n, d) containing the dataset
        * n is the number of data points.
        * d is the number of dimensions for each data point.
    - k is a positive integer containing the number of clusters.
    - iterations is a positive integer containing the maximum
    number of iterations that should be performed.
    - Returns: C, clss, or None, None on failure.
        * C is a numpy.ndarray of shape (k, d) containing the
        centroid means for each cluster.
        * clss is a numpy.ndarray of shape (n,) containing the
        index of the cluster in C that each data point belongs to.
    """
    # (k, d)
    if type(iterations) is not int or iterations < 1:
        return None, None
    C = initialize(X, k)
    if C is None:
        return None, None

    k, d = C.shape
    n, _ = X.shape

    C = np.zeros((k, d))
    clss = np.zeros((n,))
    clusters = np.empty(shape=[k, d])
    # print("k", k, " | d", d, " | n", n)
    # create clusters
    for i in range(iterations):
        centroids = np.copy(C)
        distance = np.sqrt(np.sum((X[:, None] - C)**2, axis=-1))
        clss = np.argmin(distance, axis=1)
        # print("closest", clss.shape)
        for j in range(k):
            idx = np.argwhere(clss == j)
            if len(idx) == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[idx], axis=0)
        if (centroids == C).all():
            return C, clss
    distance = np.sqrt(np.sum((X[:, None] - C)**2, axis=-1))
    clss = np.argmin(distance, axis=1)

    return C, clss
