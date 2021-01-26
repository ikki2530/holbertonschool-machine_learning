#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    - Converts a label vector into a one-hot matrix.
    - last dimension of the one-hot matrix must
    be the number of classes.
    """
    cat = K.utils.to_categorical(labels, num_classes=classes)
    return cat
