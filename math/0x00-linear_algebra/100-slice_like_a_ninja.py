#!/usr/bin/env python3
"""Slicing over a specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slice along specific axis"""
    slc = [slice(None)] * len(matrix.shape)
    for k, v in axes.items():
        # print("key:",k, "-- value:", v, ", iteración #",
        # i, "  tuple length:", len(v))
        ax = k
        if len(v) == 1:
            slc[ax] = slice(v[0])
        elif len(v) == 2:
            slc[ax] = slice(v[0], v[1])
        elif len(v) == 3:
            slc[ax] = slice(v[0], v[1], v[2])
        m = matrix[tuple(slc)]
    return m
