#!/usr/bin/env python3
"""Concatenate 2 matrices"""


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


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Description: concatenates two matrices along a specific axis,
    mat1 and mat2 are 2D matrices containing ints/floats.
    Returns: A new lists with mat1 and mat2 concatenated
    """
    # mat = []
    # if axis == 0:
    #     shape1 = matrix_shape(mat1)
    #     shape2 = matrix_shape(mat2)
    #     if shape1[1] == shape2[1]:
    #         # copy
    #         for j in range(len(mat1)):
    #             mat.append(mat1[j].copy())
    #         for i in range(len(mat2)):
    #             mat.append(mat2[i].copy())
    #         return mat
    #     else:
    #         return None
    # elif axis == 1:
    #     shape1 = matrix_shape(mat1)
    #     shape2 = matrix_shape(mat2)
    #     if shape1[0] == shape2[0]:
    #         # copy
    #         for j in range(len(mat1)):
    #             mat.append(mat1[j].copy())
    #         for i in range(len(mat)):
    #             mat[i] = mat[i] + mat2[i]
    #         return mat
    #     else:
    #         return None

    # copy to m1 and m2
    m1 = []
    m2 = []
    for j in range(len(mat1)):
        m1.append(mat1[j].copy())
    for i in range(len(mat2)):
        m2.append(mat2[i].copy())

    if axis == 0:
        # if num of columns of m1 = num of columns of m2
        if len(m1[0]) == len(m2[0]):
            for k in range(len(m2)):
                m1.append(m2[k])
        else:
            return None
    elif axis == 1:
        # if num of rows of m1 = num of rows
        if len(m1) == len(m2):
            for i in range(len(m2)):
                for k in range(len(m2[i])):
                    m1[i].append(m2[i][k])
        else:
            return None
    return m1
