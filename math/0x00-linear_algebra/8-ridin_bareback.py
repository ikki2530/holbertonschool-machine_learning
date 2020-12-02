#!/usr/bin/env python3
"""multiply 2 matrices"""


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


def mat_mul(mat1, mat2):
    """
    Description: performs matrix multiplication, 2D matrices
    Returns: a new matrix with the results of the multiplication
    if it is possible, None otherwise
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    suma = 0
    resultado = []
    temp = []
    if shape1[1] == shape2[0]:
        for k in range(len(mat1)):
            for i in range(len(mat2[0])):
                for j in range(len(mat1[0])):
                    suma += mat1[k][j] * mat2[j][i]
                temp.append(suma)
                suma = 0
            resultado.append(temp)
            temp = []
        return resultado
    else:
        return None
