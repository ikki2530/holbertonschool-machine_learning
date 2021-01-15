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
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=w)
    z = model(prev)

    gamma = tf.Variable(tf.ones((z.get_shape()[-1])))
    beta = tf.Variable(tf.zeros((z.get_shape()[-1])))

    m, v = tf.nn.moments(z, [0])
    z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, 1e-8)

    ypred = activation(z_norm)
    return ypred
