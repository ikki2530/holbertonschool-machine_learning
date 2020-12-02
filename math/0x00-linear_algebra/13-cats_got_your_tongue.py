#!/usr/bin/env python3
"""Concats 2 matrices"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Description: Concatenates 2 matrices along axis(columns=0, rows=1)
    Returns: new matrix with mat1 and mat2 concatenated
    """
    return np.concatenate((mat1, mat2), axis=axis)
