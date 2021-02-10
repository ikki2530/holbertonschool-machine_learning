#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    - Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).
    - Returns: the keras model
    """
    kernel_norm = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    # X = K.layers.ZeroPadding2D((3, 3))(X_input)

    # stage 1
    conv_1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same',
                             kernel_initializer=kernel_norm)(X)

    batch1 = K.layers.BatchNormalization(axis=3)(conv_1)

    act1 = K.layers.Activation('relu')(batch1)

    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same')(act1)

    # stage 2
    conv_block1 = projection_block(maxpool1, [64, 64, 256], 1)

    id_block1 = identity_block(conv_block1, [64, 64, 256])
    id_block1_2 = identity_block(id_block1, [64, 64, 256])

    # stage 3
    conv_block2 = projection_block(id_block1_2, [128, 128, 512], 2)

    id_block2 = identity_block(conv_block2, [128, 128, 512])
    id_block2_1 = identity_block(id_block2, [128, 128, 512])
    id_block2_2 = identity_block(id_block2_1, [128, 128, 512])

    # stage 4
    conv_block3 = projection_block(id_block2_2, [256, 256, 1024], 2)

    id_block3 = identity_block(conv_block3, [256, 256, 1024])
    id_block3_1 = identity_block(id_block3, [256, 256, 1024])
    id_block3_2 = identity_block(id_block3_1, [256, 256, 1024])
    id_block3_3 = identity_block(id_block3_2, [256, 256, 1024])
    id_block3_4 = identity_block(id_block3_3, [256, 256, 1024])

    # stage 5
    conv_block4 = projection_block(id_block3_4, [512, 512, 2048], 2)
    id_block4 = identity_block(conv_block4, [512, 512, 2048])
    id_block4_1 = identity_block(id_block4, [512, 512, 2048])

    # last
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(id_block4_1)
    flat = K.layers.Flatten()(avg_pool)
    full = K.layers.Dense(1000, activation="softmax",
                          kernel_initializer=kernel_norm)(flat)
    model = K.models.Model(inputs=X, outputs=full)

    return model
