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


def distances(a, b):
    """
    Function that computes the euclidian distance
    from all points in a which every other in b.
    - a ndarray of shape(n, d).
    - b ndarray of shape(m, d).
        - n and m are the number of points.
        - d is the dimention of each point.
    Returns: (n, m) ndarray with distances.
    """
    b = b[np.newaxis, :]
    a = a[:, np.newaxis, :]

    diff = a - b
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)

    return dist


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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    C = initialize(X, k)
    if C is None:
        return None, None

    k, d = C.shape
    n, _ = X.shape
    dist = np.zeros(n)

    for i in range(iterations):
        dists = distances(X, C)
        clss = np.argmin(dists, axis=1).reshape(-1)

        C_changed = False
        for j in range(k):
            clust = X[clss == j]
            if len(clust) == 0:
                mean_j = initialize(X, 1)[0]
            else:
                mean_j = clust.mean(axis=0)

            if (mean_j != C[j]).all():
                C[j] = mean_j
                C_changed = True

        if not C_changed:
            break

    return C, clss
