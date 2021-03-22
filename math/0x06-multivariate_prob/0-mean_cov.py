#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.
    - X is a numpy.ndarray of shape (n, d) containing the data set.
        - n is the number of data points.
        - d is the number of dimensions in each data point.
    Returns: mean, cov.
        - mean is a numpy.ndarray of shape (1, d) containing
        the mean of the data set.
        - cov is a numpy.ndarray of shape (d, d) containing
        the covariance matrix of the data set.
    """
    if type(X) is not np.ndarray and len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, _ = X.shape
    if n < 2:
        ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.dot((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
