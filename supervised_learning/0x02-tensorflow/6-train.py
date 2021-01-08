#!/usr/bin/env python3
"""Trainning"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    - Builds, trains, and saves a neural network classifier
    - X_train is a numpy.ndarray containing the training input data.
    - Y_train is a numpy.ndarray containing the training labels.
    - X_valid is a numpy.ndarray containing the validation input data
    - Y_valid is a numpy.ndarray containing the validation labels.
    - layer_sizes is a list containing the number of nodes in each
    layer of the network.
    - activations is a list containing the activation functions for
    each layer of the network.
    - alpha is the learning rate.
    - iterations is the number of iterations to train over.
    - save_path designates where to save the model.
    """
    # arquitectura, sin datos
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    # initializer, cuando no se inicializa las
    # matrices de peso esto inicia los valores
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        for i in range(iterations):
            # entrenamiento
            accur_train, loss_value_train = sess.run((
                accuracy, loss), feed_dict={x: X_train, y: Y_train})
            # validación
            accur_val, loss_value_val = sess.run((
                accuracy, loss), feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_value_train))
                print("\tTraining Accuracy: {}".format(accur_train))
                print("\tValidation Cost: {}".format(loss_value_val))
                print("\tValidation Accuracy: {}".format(accur_val))

        saver.save(sess, save_path)
    return save_path
