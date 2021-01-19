#!/usr/bin/env python3
"""
calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    - calculates the specificity for each class in a confusion matrix.
    - confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels.
    - classes is the number of classes.
    Returns: a numpy.ndarray of shape (classes,) containing
    the specificity of each class.
    """
    TPi = np.diag(confusion)
    FNi = np.sum(confusion, axis=1) - TPi
    FPi = np.sum(confusion, axis=0) - TPi
    TNi = np.sum(confusion) - (FPi + FNi + TPi)
    SPEC = TNi / (FPi + TNi)

    return SPEC
