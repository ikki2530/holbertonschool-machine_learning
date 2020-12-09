#!/usr/bin/env python3
"""Calculates i squared"""
import numpy as np


def summation_i_squared(n):
    """
    Description: calculates the square sum of the n numbers 0*0 + 1*1 + ...+n*n
    n: Limit of the numbers
    Returns: sum the n square numbers
    """
    suma = 0
    if type(n) == int:
        nums = np.arange(1, n+1)
        suma = np.dot(nums.T, nums)
    else:
        return None
    return suma
