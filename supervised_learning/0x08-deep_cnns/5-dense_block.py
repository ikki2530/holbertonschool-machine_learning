#!/usr/bin/env python3
"""
builds a dense block as described in
Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    - Builds a dense block as described in
    Densely Connected Convolutional Networks.
    - X is the output from the previous layer.
    - nb_filters is an integer representing the number of filters in X.
    - growth_rate is the growth rate for the dense block.
    - layers is the number of layers in the dense block.
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively.
    """
    kernel_norm = K.initializers.he_normal()
    X_input = X
    num_filters = nb_filters
    for i in range(layers):
        # H
        batch = K.layers.BatchNormalization()(X_input)
        activation = K.layers.Activation('relu')(batch)
        conv = K.layers.Conv2D(4 * growth_rate, kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=kernel_norm)(activation)

        batch2 = K.layers.BatchNormalization()(conv)
        activation2 = K.layers.Activation('relu')(batch2)
        conv2 = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=kernel_norm)(activation2)

        # concat
        X_input = K.layers.concatenate([X_input, conv2])
        nb_filters += growth_rate

    return X_input, nb_filters
