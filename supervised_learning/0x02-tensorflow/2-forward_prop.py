#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    - Creates the forward propagation graph for the neural network.
    - x is the placeholder for the input data.
    - layer_sizes is a list containing the number of nodes in each
    layer of the network.
    - activations is a list containing the activation functions for
    each layer of the network.
    - Returns: the prediction of the network in tensor form.
    """
    new_input = x
    for i in range(len(layer_sizes)):
        outp = create_layer(new_input, layer_sizes[i], activations[i])
        new_input = outp

    return new_input
