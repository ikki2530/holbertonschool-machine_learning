#!/usr/bin/env python3
"""
calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    calculates a correlation matrix
    - C is a numpy.ndarray of shape (d, d) containing a covariance matrix.
        -d is the number of dimensions
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix.
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    d1, d2 = C.shape
    if len(C.shape) != 2 or d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    diagonal = np.sqrt(np.diag(C)).reshape((C.shape[0], 1))
    inv_diag = 1 / (diagonal @ diagonal.T)

    cor_m = inv_diag * C

    return cor_m
