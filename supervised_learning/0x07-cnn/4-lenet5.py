#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.
    - x is a tf.placeholder of shape (m, 28, 28, 1) containing the
    input images for the network, m is the number of images.
    - y is a tf.placeholder of shape (m, 10) containing
    the one-hot labels for the network.
    """
    hnorm = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=hnorm)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=hnorm)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    flat = tf.layers.Flatten()(pool2)

    full1 = tf.layers.Dense(units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=hnorm)(flat)

    full2 = tf.layers.Dense(units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=hnorm)(full1)

    outp = tf.layers.Dense(units=10, kernel_initializer=hnorm)(full2)

    y_hat = tf.nn.softmax(outp)
    y_hat_tag = tf.argmax(outp, 1)
    prediction = tf.argmax(y, 1)
    acc = tf.equal(y_hat_tag, prediction)
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, outp)
    trainer = tf.train.AdamOptimizer().minimize(loss)

    return y_hat, trainer, loss, accuracy
