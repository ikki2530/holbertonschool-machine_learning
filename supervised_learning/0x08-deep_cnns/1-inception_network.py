#!/usr/bin/env python3
"""
builds the inception network as described
in Going Deeper with Convolutions (2014).
"""
import tensorflow.keras as K


inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described
    in Going Deeper with Convolutions (2014).
    Returns: the keras model
    """
    kernel_norm = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', activation='relu',
                             kernel_initializer=kernel_norm)(X)
    l1_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(layer1)
    # 3
    layer2_1 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer=kernel_norm)(l1_pool)

    layer2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                             strides=(1, 1), padding='same',
                             activation='relu',
                             kernel_initializer=kernel_norm)(layer2_1)

    l2_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(layer2)
    # inception 1
    inception1 = inception_block(l2_pool, [64, 96, 128, 16, 32, 32])

    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    # 3 pool
    l3_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(inception2)

    # inception3
    inception3 = inception_block(l3_pool, [192, 96, 208, 16, 48, 64])

    # inception 4
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])

    # inception 5
    inception5 = inception_block(inception4, [128, 128, 256, 24, 64, 64])

    # inception 6
    inception6 = inception_block(inception5, [112, 144, 288, 32, 64, 64])

    # inception 7
    inception7 = inception_block(inception6, [256, 160, 320, 32, 128, 128])

    # 4 pool
    l4_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(inception7)

    # inception 8
    inception8 = inception_block(l4_pool, [256, 160, 320, 32, 128, 128])

    # inception 9
    inception9 = inception_block(inception8, [384, 192, 384, 48, 128, 128])

    # avg pooling 1
    avg_pool1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                          strides=(7, 7),
                                          padding="same")(inception9)

    # dropout 1
    dropout1 = K.layers.Dropout(0.4)(avg_pool1)

    # dense 1
    dense1 = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=kernel_norm)(dropout1)

    model = K.models.Model(inputs=X, outputs=dense1)

    return model
