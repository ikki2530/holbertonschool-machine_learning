#!/usr/bin/env python3
"""
Calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix.
    - matrix is a numpy.ndarray of shape (n, n)
    whose definiteness should be calculated.
    - Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite if
    the matrix is positive definite, positive semi-definite,
    negative semi-definite, negative definite of indefinite, respectively.
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    dims = matrix.shape
    if len(dims) != 2:
        return None
    if dims[0] != dims[1]:
        return None

    eigvalues, _ = np.linalg.eig(matrix)
    print("eigenvalues", eigvalues)

    semipositive = np.where(eigvalues >= 0, True, False)
    seminegative = np.where(eigvalues <= 0, True, False)

    if False not in semipositive:
        positive = np.where(eigvalues > 0, True, False)
        if False not in positive:
            return "Positive definite"
        return "Positive semi-definite"
    elif False not in seminegative:
        negative = np.where(eigvalues < 0, True, False)
        if False not in negative:
            return "Negative definite"
        return "Negative semi-definite"
    elif False in semipositive and True in semipositive:
        return "Indefinite"
