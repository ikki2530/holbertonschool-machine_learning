#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    - labels is a one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point,
    m is the number of data points and classes is the number of classes.
    - logits is a one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels.
    """
    conf = labels.T @ logits
    return conf
