#!/usr/bin/env python3


def matrix_shape(matrix):
    """
    matrix: matrix to calcuted the shape
    Return: A list with the matrix shape [n, m], n is the number of rows and m number of columns
    """
    lista = []
    if type(matrix) == list:
        dm = len(matrix)
        lista.append(dm)
        lista = lista + matrix_shape(matrix[0])
        return lista
    else:
        return lista
