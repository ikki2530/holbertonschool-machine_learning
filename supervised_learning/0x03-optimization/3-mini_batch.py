#!/usr/bin/env python3
"""Normalize"""
import numpy as np
import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_batch(X, batch_size):
    """Function to create batches from a data set."""
    m = X.shape[0]
    n_batches = int(m / batch_size)

    batches_list = []
    for i in range(0, n_batches):
        a = i * batch_size
        b = a + batch_size
        X_mini = X[a:b]
        batches_list.append(X_mini)

    if m % batch_size != 0:
        r = m % batch_size

        a = n_batches * batch_size
        b = a + r
        X_mini = X[a:b]
        batches_list.append(X_mini)

    return batches_list


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    - Trains a loaded neural network model using mini-batch gradient descent.
    - Y_train is a one-hot numpy.ndarray of shape (m, 10) containing
    the training labels.
    - X_valid is a numpy.ndarray of shape (m, 784) containing the
    validation data.
    - Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing
    the validation labels.
    - batch_size is the number of data points in a batch.
    - epochs is the number of times the training should pass through
    the whole dataset.
    - load_path is the path from which to load the model.
    - save_path is the path to where the model should be saved after training.
    """
    with tf.Session() as sess:
        saved = tf.train.import_meta_graph("{}.meta".format(load_path))
        saved.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        for i in range(epochs + 1):
            accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                feed_dict={x: X_train,
                                                           y: Y_train})
            accuracy_v, loss_value_v = sess.run((accuracy, loss),
                                                feed_dict={x: X_valid,
                                                           y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_value_t))
            print("\tTraining Accuracy: {}".format(accuracy_t))
            print("\tValidation Cost: {}".format(loss_value_v))
            print("\tValidation Accuracy: {}".format(accuracy_v))
            if i < epochs:
                X, Y = shuffle_data(X_train, Y_train)
                batches_x = create_batch(X, batch_size)
                batches_y = create_batch(Y, batch_size)
                for i, b_x in enumerate(batches_x):
                    b_y = batches_y[i]
                    sess.run(train_op, feed_dict={x: b_x,
                                                  y: b_y})
                    if (i + 1) % 100 == 0 and i != 0:
                        accuracy_t, loss_value_t = sess.run((accuracy, loss),
                                                            feed_dict={x: b_x,
                                                                       y: b_y})
                        print("\tStep {}".format(i + 1))
                        print("\t\tCost: {}".format(loss_value_t))
                        print("\t\tAccuracy: {}".format(accuracy_t))

        save_path = saved.save(sess, save_path)
        return save_path
