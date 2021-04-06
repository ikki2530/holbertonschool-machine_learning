#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain.
    - P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the transition matrix.
        * P[i, j] is the probability of transitioning from state i to state j.
        * n is the number of states in the markov chain.
    - Returns: a numpy.ndarray of shape (1, n) containing the
    steady state probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    s = np.ones((1, n1)) / n1
    p_copy1 = P.copy()

    if np.any(p_copy1 <= 0):
        return None
    while True:
        s_copy = s.copy()
        s = np.matmul(s, P)
        if np.all(s_copy == s):
            return s
