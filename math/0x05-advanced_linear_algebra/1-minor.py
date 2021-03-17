#!/usr/bin/env python3
"""
Calculates the minor matrix of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    - matrix is a list of lists whose determinant should be calculated.
    Returns: the determinant of matrix
    """
    n = len(matrix)
    if n == 1 and len(matrix[0]) == 0 and type(
       matrix) == list and type(matrix[0]) == list:
        return 1
    if n == 0:
        raise TypeError("matrix must be a list of lists")

    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if type(row) != list:
            raise TypeError("matrix must be a list of lists")

        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = a * d - (b * c)
        return det
    all_minors = []
    mult = matrix[0]
    signo = 1
    signos = []
    newm = []
    temp = []
    cofactorv = 0
    # take the minors
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != cofactorv and j != k:
                    temp.append(matrix[i][j])
            if temp:
                newm.append(temp.copy())
            temp = []
        if newm:
            all_minors.append(newm)
        signos.append(signo)
        signo = signo * -1
        newm = []
    # add determinant
    suma = 0
    for i in range(n):
        suma = suma + (signos[i] * mult[i] * determinant(all_minors[i]))
    return suma


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    - matrix is a list of lists whose minor matrix should be calculated.
    Returns: the minor matrix of matrix
    """
    n = len(matrix)
    if n == 1 and len(matrix[0]) == 0 and type(
       matrix) == list and type(matrix[0]) == list:
        return [[0]]

    if n == 1 and len(matrix[0]) == 1 and type(
       matrix) == list and type(matrix[0]) == list:
        return [[1]]

    if n == 0:
        raise TypeError("matrix must be a list of lists")

    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if type(row) != list:
            raise TypeError("matrix must be a list of lists")

        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    all_minors = []
    row_minors = []
    newm = []
    temp = []
    # find the minor matrices
    for h in range(n):
        for w in range(n):
            for i in range(n):
                for j in range(n):
                    if i != h and j != w:
                        temp.append(matrix[i][j])
                if temp:
                    newm.append(temp.copy())
                temp = []
            if newm:
                # Add a new minor
                all_minors.append(newm)
            newm = []
        if all_minors:
            row_minors.append(all_minors.copy())
        all_minors = []  # clean for the next row
    # find the determinant of each minor matrix
    minors = []
    temp2 = []
    for list_minors in row_minors:
        for m in list_minors:
            det = determinant(m)
            temp2.append(det)
        if temp2:
            minors.append(temp2.copy())
        temp2 = []
    return minors
