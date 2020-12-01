#!/usr/bin/env python3


def np_elementwise(mat1, mat2):
    """
    Description: Calculates the sum, difference, product, and quotient
    Returns: A tuple with the sum, difference, product, and quotient
    respectively
    """
    suma = mat1 + mat2
    resta = mat1 - mat2
    mult = mat1 * mat2
    div = mat1 / mat2
    return (suma, resta, mult, div)
