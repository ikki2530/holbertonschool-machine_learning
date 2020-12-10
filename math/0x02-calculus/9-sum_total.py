#!/usr/bin/env python3
"""Calculates i squared"""


def summation_i_squared(n):
    """
    Description: calculates the square sum of the n numbers 0*0 + 1*1 + ...+n*n
    n: Limit of the numbers
    Returns: sum the n square numbers
    """
    suma = 0
    if (type(n) == int or type(n) == float) and n >= 0:
        suma = ((n ** 3) / 3) + ((n ** 2)/2) + (n/6)
        if suma % 1 == 0:
            suma = int(suma)
    else:
        return None
    return suma
