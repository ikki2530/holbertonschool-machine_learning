#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture as described in
Densely Connected Convolutional Networks.
"""
import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as
    described in Densely Connected Convolutional Networks.
    - growth_rate is the growth rate.
    - compression is the compression factor.
    Returns: the keras model.
    """
    kernel_norm = K.initializers.he_normal()

    X = K.Input(shape=(224, 224, 3))

    batch = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch)
    conv = K.layers.Conv2D(2 * growth_rate, kernel_size=(7, 7), strides=(2, 2),
                           padding='same',
                           kernel_initializer=kernel_norm)(activation)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding="same")(conv)

    dense_b, n_filters = dense_block(max_pool, 2*growth_rate, growth_rate, 6)

    transition, n_filters1 = transition_layer(dense_b, n_filters, compression)

    dense_b1, n_filters2 = dense_block(transition, n_filters1, growth_rate, 12)

    transition1, n_filters3 = transition_layer(dense_b1, n_filters2,
                                               compression)

    dense_b2, n_filters4 = dense_block(transition1, n_filters3,
                                       growth_rate, 24)

    transition2, n_filters5 = transition_layer(dense_b2, n_filters4,
                                               compression)

    dense_b3, n_filters6 = dense_block(transition2, n_filters5,
                                       growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=7,
                                         padding="same")(dense_b3)

    output = K.layers.Dense(1000, activation="softmax",
                            kernel_initializer=kernel_norm)(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)

    return model
