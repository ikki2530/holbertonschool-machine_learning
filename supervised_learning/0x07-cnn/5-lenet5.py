#!/usr/bin/env python3
"""
builds a modified version of the LeNet-5 architecture using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.
    - X is a K.Input of shape (m, 28, 28, 1) containing the
    input images for the network, m is the number of images.
    """
    h_ini = K.initializers.he_normal(seed=None)
    # model = K.Sequential()
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                            activation='relu', kernel_initializer=h_ini)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                            activation='relu', kernel_initializer=h_ini)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)
    flat = K.layers.Flatten()(pool2)
    dns1 = K.layers.Dense(units=120, activation='relu',
                          kernel_initializer=h_ini)(flat)
    dns2 = K.layers.Dense(units=84, activation='relu',
                          kernel_initializer=h_ini)(dns1)
    outp = K.layers.Dense(units=10, activation='softmax',
                          kernel_initializer=h_ini)(dns2)

    model = K.Model(inputs=X, outputs=outp)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
