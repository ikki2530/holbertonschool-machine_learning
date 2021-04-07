#!/usr/bin/env python3
"""Determines if a markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing.
    - P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix.
        * P[i, j] is the probability of transitioning from state i to state j.
        * n is the number of states in the markov chain.
    Returns: True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    d = np.diag(P)
    if (d == 1).all():
        return True
    if not (d == 1).any():
        return False

    for i in range(n1):
        for j in range(n2):
            if i == j:
                if P[i][j] == 1:
                    return True
    return False
