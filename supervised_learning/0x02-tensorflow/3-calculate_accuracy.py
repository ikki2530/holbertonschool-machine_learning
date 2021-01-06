#!/usr/bin/env python3
"""Calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    - Calculates the accuracy of a prediction.
    - y is a placeholder for the labels of the input data.
    - y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction.
    """
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
