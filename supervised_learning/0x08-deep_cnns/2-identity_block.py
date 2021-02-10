#!/usr/bin/env python3
"""
builds an identity block as described in
Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    - Builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015).
    - A_prev is the output from the previous layer.
    - filters is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution.
        - F3 is the number of filters in the 3x3 convolution.
        - F12 is the number of filters in the second 1x1 convolution.
    Returns: the activated output of the identity block.
    """
    kernel_norm = K.initializers.he_normal()
    F11, F3, F12 = filters
    # 1
    conv_1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_norm)(A_prev)

    batch1 = K.layers.BatchNormalization(axis=3)(conv_1)

    act1 = K.layers.Activation('relu')(batch1)

    # 2
    conv_2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_norm)(act1)

    batch2 = K.layers.BatchNormalization(axis=3)(conv_2)

    act2 = K.layers.Activation('relu')(batch2)

    # 3
    conv_3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_norm)(act2)

    batch3 = K.layers.BatchNormalization(axis=3)(conv_3)

    # act3 = K.layers.Activation('relu')(batch3)

    added = K.layers.Add()([batch3, A_prev])

    outp = K.layers.Activation('relu')(added)
    return outp
