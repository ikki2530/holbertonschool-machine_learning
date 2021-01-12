#!/usr/bin/env python3
"""Normalize constants"""
import numpy as np


def normalize(X, m, s):
    """
    - X is the numpy.ndarray of shape (d, nx) to normalize,
    d is the number of data points and nx is the number of features.
    - m is a numpy.ndarray of shape (nx,) that contains the mean
    of all features of X.
    - s is a numpy.ndarray of shape (nx,) that contains the standard
    deviation of all features of X.
    - Returns: The normalized X matrix.
    """
    X_norm = (X - m.T) / s.T
    return X_norm
