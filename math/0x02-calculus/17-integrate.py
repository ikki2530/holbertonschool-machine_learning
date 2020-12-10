#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    - Description: calculates the integral of a polynomial
    - poly: is a list of coefficients representing a polynomial
    - C is an integer representing the integration constant
    - Returns: new list of coefficients representing the integral
    of the polynomial
    """
    if type(C) == int and len(poly) != 0 and type(poly) == list:
        lg = len(poly)
        new_coef = []
        new_coef.append(C)

        for i in range(lg):
            if type(poly[i]) != int and type(poly[i]) != float:
                return None
            coef = poly[i]
            grade = i
            new_coef.append(coef/(grade + 1))
            if new_coef[i + 1] % 1 == 0 and type(new_coef[i + 1]) == float:
                new_coef[i + 1] = int(new_coef[i + 1])
    else:
        return None
    return new_coef
