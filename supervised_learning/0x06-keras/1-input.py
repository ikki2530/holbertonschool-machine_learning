#!/usr/bin/env python3
"""
builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    - builds a neural network with the Keras library.
    - nx is the number of input features to the network.
    - layers is a list containing the number of nodes
    in each layer of the network.
    - activations is a list containing the activation
    functions used for each layer of the network.
    - lambtha is the L2 regularization parameter.
    - keep_prob is the probability that a node will be kept for dropout.
    """
    inp = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    y = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=reg, name="dense")(inp)
    for i in range(1, len(activations)):
        inp_next = K.layers.Dropout(1 - keep_prob)(y)
        y = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=reg,
                           name="dense_" + str(i))(inp_next)
    model = K.Model(inputs=inp, outputs=y)

    return model
