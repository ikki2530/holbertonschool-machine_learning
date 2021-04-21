#!/usr/bin/env python3
"""Function that creates an autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder.
    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively.
    - latent_dims is an integer containing the dimensions
    of the latent space representation.
    - Returns: encoder, decoder, auto.
        * encoder is the encoder model.
        * decoder is the decoder model.
        * auto is the full autoencoder model.
    """
    input_x = keras.layers.Input(shape=(input_dims,))
    encode = keras.layers.Dense(hidden_layers[0], activation='relu')(input_x)
    for i in range(1, len(hidden_layers)):
        encode = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(encode)
    encode = keras.layers.Dense(latent_dims, activation='relu')(encode)
    decode = keras.layers.Input(shape=(latent_dims,))
    input_decoder = decode
    for i in range(len(hidden_layers) - 1, -1, -1):
        decode = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(decode)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    encoder = keras.models.Model(input_x, encode)
    decoder = keras.models.Model(input_decoder, decode)
    auto = keras.models.Model(input_x, decoder(encoder(input_x)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
