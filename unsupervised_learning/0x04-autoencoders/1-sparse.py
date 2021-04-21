#!/usr/bin/env python3
"""Creates a sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder.
    - input_dims is an integer containing the dimensions of the model input.
    - hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively.
        * the hidden layers should be reversed for the decoder.
    - latent_dims is an integer containing the dimensions of the latent
    space representation.
    - lambtha is the regularization parameter used for L1 regularization
    on the encoded output.
    - Returns: encoder, decoder, auto
        * encoder is the encoder model.
        * decoder is the decoder model.
        * auto is the sparse autoencoder model.
    """
    # https://blog.keras.io/building-autoencoders-in-keras.html
    input_x = keras.layers.Input(shape=(input_dims,))
    encode = keras.layers.Dense(hidden_layers[0], activation='relu')(input_x)
    for i in range(1, len(hidden_layers)):
        encode = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(encode)
    reg = keras.regularizers.l1(lambtha)
    encode_output = keras.layers.Dense(latent_dims,
                                       activation='relu',
                                       activity_regularizer=reg)(encode)
    decode = keras.layers.Input(shape=(latent_dims,))
    input_decoder = decode
    for i in range(len(hidden_layers) - 1, -1, -1):
        decode = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(decode)
    decode_output = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(decode)
    encoder = keras.models.Model(input_x, encode_output)
    decoder = keras.models.Model(input_decoder, decode_output)
    auto = keras.models.Model(input_x, decoder(encoder(input_x)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
