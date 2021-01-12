#!/usr/bin/env python3
"""Normalize constants"""
import numpy as np


def normalization_constants(X):
    """
    - X is the numpy.ndarray of shape (m, nx) to normalize,
    m is the number of data points and nx is the number of features.
    """
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m  # 1 x nx
    var = np.sum((X - mean)**2, axis=0) / m
    std = np.sqrt(var)
    return mean, std
