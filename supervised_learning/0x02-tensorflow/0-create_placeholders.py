#!/usr/bin/env python3
"""placeholders x and y"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    - Returns two placeholders, x and y,
    for the neural network.
    - nx: the number of feature columns in our data.
    - classes: the number of classes in our classifier.
    """
    x1 = tf.placeholder("float", (None, nx), name="x")
    y1 = tf.placeholder("float", (None, classes), name="y")

    return x1, y1
