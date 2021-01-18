#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    - Calculates the sensitivity for each class in a confusion matrix.
    - confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels.
    - classes is the number of classes
    """
    total_rows = np.sum(confusion, axis=1)
    diag = confusion.diagonal()
    sensiv = diag / total_rows
    return sensiv
