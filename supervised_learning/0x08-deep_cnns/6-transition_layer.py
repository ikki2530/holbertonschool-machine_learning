#!/usr/bin/env python3
"""
builds a transition layer as described in
Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    - Builds a transition layer as described in
    Densely Connected Convolutional Networks.
    - X is the output from the previous layer.
    - nb_filters is an integer representing the number of filters in X.
    - compression is the compression factor for the transition layer.
    Returns: The output of the transition layer and the number of
    filters within the output, respectively.
    """
    kernel_norm = K.initializers.he_normal()
    batch = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch)
    m = X.shape[3].value
    filters = int(compression * m)
    conv = K.layers.Conv2D(filters, kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=kernel_norm)(activation)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2))(conv)

    return avg_pool, filters
