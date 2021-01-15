#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    - Creates a batch normalization layer for a neural network in tensorflow.
    - prev is the activated output of the previous layer.
    - n is the number of nodes in the layer to be created.
    - activation is the activation function that should
    be used on the output of the layer.
    - you should use an epsilon of 1e-8.
    Returns: a tensor of the activated output for the layer
    """

