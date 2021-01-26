#!/usr/bin/env python3
"""
makes a prediction using a neural network.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    - Makes a prediction using a neural network
    - network is the network model to make the prediction with.
    - data is the input data to make the prediction with.
    - verbose is a boolean that determines if output should
    be printed during the prediction process.
    """
    # https://www.machinecurve.com/index.php/2020/02/21/
    # how-to-predict-new-samples-with-your-keras-model/
    # https://www.educative.io/edpresso/what-is-model-predict-in-keras
    prediction = network.predict(data, verbose=verbose)
    return prediction
