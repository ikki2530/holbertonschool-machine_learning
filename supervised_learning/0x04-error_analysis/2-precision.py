#!/usr/bin/env python3
"""
calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    - Calculates the precision for each class in a confusion matrix.
    - confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels, classes is the number of classes.
    Returns: a numpy.ndarray of shape (classes,) containing
    the precision of each class.
    """
    total_colums = np.sum(confusion, axis=0)
    diag = confusion.diagonal()
    precs = diag / total_colums
    return precs
