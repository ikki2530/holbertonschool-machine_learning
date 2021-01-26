#!/usr/bin/env python3
"""
Saves an entire model.
Loads an entire model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    - Saves an entire model.
    - network is the model to save.
    - filename is the path of the file that the model should be saved to.
    Returns: None.
    """
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """
    - Loads an entire model.
    - filename is the path of the file that the model should be loaded from.
    Returns: the loaded model.
    """
    return K.models.load_model(filepath=filename)
