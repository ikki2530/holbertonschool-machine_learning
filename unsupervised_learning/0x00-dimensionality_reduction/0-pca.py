#!/usr/bin/env python3
"""performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    """
    # cov = np.cov(X.T)
    # # print("cov", cov, " | shape", cov.shape)
    # eig_val, eig_vect = np.linalg.eig(cov)
    # # order eigenvalues
    # idx = eig_val.argsort()[::-1]
    # eig_val = eig_val[idx]
    # eig_vect = eig_vect[:, idx]
    # # eig_vect = eig_vect[:, 0:3]
    # # print("eig vectors", eig_vect)

    # ratios = eig_val / np.sum(eig_val)
    # variance = np.cumsum(ratios)
    # r = np.argwhere(variance >= var)
    # print("variance cumulative", variance)
    # # print("ratio", ratios)
    # r = r[0, 0]
    # w = eig_vect[:, r + 1]
    # return w
    _, S, V = np.linalg.svd(X)
    idx = S.argsort()[::-1]
    eig_val = S[idx]
    eig_vect = V.T
    eig_vect = eig_vect[:, idx]

    ratios = eig_val / np.sum(eig_val)
    variance = np.cumsum(ratios)
    r = np.argwhere(variance >= var)[0, 0]
    w = eig_vect[:, 0:r+1]
    return w
