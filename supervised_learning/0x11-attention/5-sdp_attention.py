#!/usr/bin/env python3
"""Function that calculates the scaled dot product attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Function that calculates the scaled dot product attention"""
    mat_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention = mat_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += (mask * -1e9)
    weights = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
