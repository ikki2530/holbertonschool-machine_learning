#!/usr/bin/env python3
"""
train the model with learning rate decay.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
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
    - early_stopping is a boolean that indicates whether early stopping
    should be used.
    - patience is the patience used for early stopping.
    - learning_rate_decay is a boolean that indicates whether learning
    rate decay should be used.
    - verbose is a boolean that determines if output should be printed
    during training.
    - shuffle is a boolean that determines whether to shuffle the batches
    every epoch.
    """
    # https://dzlab.github.io/dltips/en/keras/learning-rate-scheduler/
    def scheduler(epoch):
        """
        - epochs is the number of passes through data for mini-batch
        gradient descent.
        - alpha is the initial learning rate
        """
        lr = alpha / (1 + (epoch*decay_rate))
        return lr
    early = []
    if early_stopping and validation_data:
        early.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience))
    if learning_rate_decay and validation_data:
        early.append(K.callbacks.LearningRateScheduler(scheduler, verbose=1))

    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          shuffle=shuffle, verbose=verbose, callbacks=early)
    return history
