#!/usr/bin/env python3
"""add 2 arrays"""


def add_arrays(arr1, arr2):
    """
    arr1: array #1 for the sum
    arr2: array # 2 for the sum
    Returns: an array with element-wise add operation of arr1 + arr2
    """
    suma = []
    if len(arr1) != len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            suma.append(arr1[i] + arr2[i])
        return suma
