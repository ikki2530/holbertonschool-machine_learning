#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    - Trains a model using mini-batch gradient descent.
    - network is the model to train.
    - data is a numpy.ndarray of shape (m, nx) containing the input data.
    - labels is a one-hot numpy.ndarray of shape (m, classes)
    containing the labels of data.
    - batch_size is the size of the batch used for mini-batch gradient
    descent.
    - epochs is the number of passes through data for mini-batch
    gradient descent.
    - validation_data is the data to validate the model with, if not None
    - verbose is a boolean that determines if output should be printed
    during training.
    - shuffle is a boolean that determines whether to shuffle the batches
    every epoch.
    """
    history = network.fit(x=data, y=labels, validation_data=validation_data,
                          epochs=epochs, verbose=verbose,
                          batch_size=batch_size, shuffle=shuffle)
    return history
