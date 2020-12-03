#!/usr/bin/env python3
"""Add 2 matricess"""


def matrix_shape(matrix):
    """
    matrix: matrix to calcuted the shape
    Return: A list with the matrix shape [n, m],
    n is the number of rows and m number of columns
    """
    lista = []
    if type(matrix) == list:
        dm = len(matrix)
        lista.append(dm)
        lista = lista + matrix_shape(matrix[0])
        return lista
    else:
        return lista


def add_matrices2D(mat1, mat2):
    """
    mat1: matrix #1 for the add operation
    mat2: matrix #2 for the add operation
    Returns: A new matrix with the adding element-wise operation of mat1 + mat2
    """
    # shape1 = matrix_shape(mat1)
    # shape2 = matrix_shape(mat2)
    # cols = []
    # mat = []
    # m1 = []
    # m2 = []
    # for row in mat1:
    #     m1.append(row.copy())
    # for row in mat2:
    #     m2.append(row.copy())

    # if shape1 == shape2:
    #     for i in range(len(m1)):
    #         for j in range(len(m1[0])):
    #             cols.append(m1[i][j] + m2[i][j])
    #         mat.append(cols)
    #         cols = []
    #     return mat
    # else:
    #     return None
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        cols = []
        resultado = []
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                cols.append(mat1[i][j] + mat2[i][j])
            resultado.append(cols)
            cols = []
        return resultado
    else:
        return None
