#!/usr/bin/env python3
"""Create Layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    - prev is the tensor output of the previous layer
    - n is the number of nodes in the layer to create.
    - activation is the activation function that the layer should use.
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=w,
                            activation=activation, name="layer")
    return model(prev)
