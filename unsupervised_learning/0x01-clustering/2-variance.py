#!/usr/bin/env python3
"""calculates the total intra-cluster variance for a data set"""
import numpy as np


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

    print("a -> x", a.shape)
    print("b -> C", b.shape)
    diff = a - b
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)

    return dist


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    - X is a numpy.ndarray of shape (n, d) containing the data set.
    - C is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster.
    - Returns: var, or None on failure.
        * var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d = X.shape
    distance = ((X - C[:, np.newaxis])**2).sum(axis=-1)
    min_distance = np.min(distance, axis=0)
    var = np.sum(min_distance)
    return var
