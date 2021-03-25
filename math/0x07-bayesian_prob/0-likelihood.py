#!/usr/bin/env python3
"""
calculates the likelihood of obtaining this data given various
hypothetical probabilities of developing severe side effects.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    - x is the number of patients that develop severe side effects
    - n is the total number of patients observed
    - P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects.

    Returns: a 1D numpy.ndarray containing the likelihood of obtaining
    the data, x and n, for each probability in P, respectively.
    """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.logical_and(P >= 0, P <= 1).all():
        raise ValueError("All values in P must be in the range [0, 1]")

    denominador = np.math.factorial(x) * np.math.factorial(n - x)
    factorial = np.math.factorial(n) / denominador

    prob = factorial * np.power(P, x) * np.power(P - 1, n - x)

    return prob
