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
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")

    denominador = np.math.factorial(x) * np.math.factorial(n - x)
    factorial = np.math.factorial(n) / denominador

    prob = factorial * np.power(P, x) * np.power(1 - P, n - x)

    return prob


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this
    data with the various hypothetical probabilities.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.ndim != 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if Pr.shape[0] != P.shape[0]:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if (P < 0).any() or (P > 1).any():
        message = "All values in P must be in the range [0, 1]"
        raise ValueError(message)
    if (Pr < 0).any() or (Pr > 1).any():
        message = "All values in Pr must be in the range [0, 1]"
        raise ValueError(message)

    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    likeh = likelihood(x, n, P)
    intersection = likeh * Pr

    return intersection
