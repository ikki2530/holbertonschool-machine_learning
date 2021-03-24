#!/usr/bin/env python3
"""performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset
    """
    X = X - np.mean(X, axis=0)
    _, S, V = np.linalg.svd(X)
    idx = S.argsort()[::-1]
    eig_val = S[idx]
    eig_vect = V.T
    eig_vect = eig_vect[:, idx]

    w = eig_vect[:, 0: ndim]
    T = np.matmul(X, w)
    return T
