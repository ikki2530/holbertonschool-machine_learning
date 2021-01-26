#!/usr/bin/env python3
"""
- saves a model’s configuration in JSON format.
- loads a model with a specific configuration.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    - Saves a model’s configuration in JSON format
    - network is the model whose configuration should be saved.
    - filename is the path of the file that the configuration
    should be saved to.
    Returns: None.
    """
    config = network.to_json()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config)
    return None


def load_config(filename):
    """
    - Loads a model with a specific configuration
    - filename is the path of the file containing
    the model’s configuration in JSON format.
    - Returns: the loaded model
    """
    with open(filename, "r", encoding="utf-8") as f:
        config = f.read()
        loaded = K.models.model_from_json(config)
    return loaded
