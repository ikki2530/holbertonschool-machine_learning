#!/usr/bin/env python3
"""Creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.
    - input_dims is a tuple of integers containing the dimensions
    of the model input.
    - filters is a list containing the number of filters for each
    convolutional layer in the encoder, respectively.
        * The filters should be reversed for the decoder
    - latent_dims is a tuple of integers containing the dimensions
    of the latent space representation.
    - Each convolution in the encoder should use a kernel size of
    (3, 3) with same padding and relu activation, followed by max
    pooling of size (2, 2).
    - Each convolution in the decoder, except for the last two,
    should use a filter size of (3, 3) with same padding and relu
    activation, followed by upsampling of size (2, 2).
        * The second to last convolution should instead use valid padding.
        * The last convolution should have the same number of filters as
        the number of channels in input_dims with sigmoid
        activation and no upsampling.
    - Returns: encoder, decoder, auto
        * encoder is the encoder model.
        * decoder is the decoder model.
        * auto is the full autoencoder model
    """
    input_x = keras.layers.Input(shape=input_dims)
    encode = input_x
    for i in range(0, len(filters)):
        encode = keras.layers.Conv2D(filters[i], (3, 3),
                                     padding="same",
                                     activation='relu')(encode)
        encode = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding='same')(encode)
    decode = keras.layers.Input(shape=latent_dims)
    input_decode = decode
    padding = 'same'
    for i in range(len(filters) - 1, 0, -1):
        if i == 0:
            padding = 'valid'
        decode = keras.layers.Conv2D(filters[i], (3, 3),
                                     padding=padding,
                                     activation='relu')(decode)
        decode = keras.layers.UpSampling2D((2, 2))(decode)
    decode = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                 padding="same",
                                 activation='sigmoid')(decode)

    encoder = keras.models.Model(input_x, encode)
    decoder = keras.models.Model(input_decode, decode)
    auto = keras.models.Model(input_x, decoder(encoder(input_x)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
