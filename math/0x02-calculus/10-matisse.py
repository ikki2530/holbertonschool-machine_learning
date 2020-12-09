#!/usr/bin/env python3
"""Derivative of a polynomial"""


def poly_derivative(poly):
    """
    -Description: calculates the derivative of a polynomial.
    the index of the list represents the power of x that the
    coefficient belongs to.
    -poly: is a list of coefficients representing a polynomial
    - Returns: new list of coefficients representing the derivative
    of the polynomial
    """
    if poly:
        new_coef = []
        lg = len(poly)
        if lg == 1:
            new_coef.append(0)
        for i in range(lg):
            if i != 0:
                coef = poly[i]
                grade = i
                new_coef.append(coef * grade)
    else:
        return None
    return new_coef
