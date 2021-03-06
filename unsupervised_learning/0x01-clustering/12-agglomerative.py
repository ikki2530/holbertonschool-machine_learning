#!/usr/bin/env python3
"""performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """performs agglomerative clustering on a dataset
    - X is a numpy.ndarray of shape (n, d) containing the dataset.
    - dist is the maximum cophenetic distance for all clusters.
    Returns: clss, a numpy.ndarray of shape (n,) containing
    the cluster indices for each data point.
    """
    linked = scipy.cluster.hierarchy.linkage(X, 'ward')
    clss = scipy.cluster.hierarchy.fcluster(linked, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linked, color_threshold=dist)
    plt.show()
    return clss
