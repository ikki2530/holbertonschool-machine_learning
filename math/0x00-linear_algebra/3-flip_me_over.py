#!/usr/bin/env python3
"""Transpose of 2D matrix"""


def matrix_transpose(matrix):
    """
    matrix: matrix to be transposed
    Returns: Transpose of matrix
    """
    m2 = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == 0:
                m2.append([])
            m2[j].append(matrix[i][j])
    return m2
