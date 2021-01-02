#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    - Converts a numeric label vector into a one-hot matrix.
    - Y is a numpy.ndarray with shape (m,) containing
    numeric class labels.
    - classes is the maximum number of classes found in Y.
    - Returns: a one-hot encoding of Y with shape (classes, m)
    or None on failure.
    """
    b = None
    if Y.size != 0:
        b = np.eye(classes)[Y].T

    return b
