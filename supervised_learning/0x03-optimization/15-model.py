#!/usr/bin/env python3
"""
builds, trains, and saves a neural network model in
tensorflow using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization.
"""
import tensorflow as tf
import numpy as np


def forward_prop(x, layer_sizes, activations, epsilon=1e-8):
    """
    Function that creates the forward propagation graph for the NN
    Arguments:
     - x: is the placeholder for the input data.
     - layer_sizes: is a list containing the number of nodes in each layer of
      the network.
     _ activations: is a list containing the activation functions for each
      layer of the network.
    Returns:
    The prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            layer = create_batch_norm_layer(x, layer_sizes[i], activations[i],
                                            epsilon=1e-8)
        else:
            layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return layer


def create_placeholders(nx, classes):
    """
    - Returns two placeholders, x and y,
    for the neural network.
    - nx: the number of feature columns in our data.
    - classes: the number of classes in our classifier.
    """
    x1 = tf.placeholder("float", (None, nx), name="x")
    y1 = tf.placeholder("float", (None, classes), name="y")

    return x1, y1


def create_layer(prev, n, activation):
    """
    - prev is the tensor output of the previous layer
    - n is the number of nodes in the layer to create.
    - activation is the activation function that the layer should use.
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=w,
                            activation=activation, name="layer")
    return model(prev)


def shuffle_data(X, Y):
    """
    - Shuffles the data points in two matrices the same way.
    - X is the first numpy.ndarray of shape (m, nx) to shuffle,
    m is the number of data points and nx is the number of features in X.
    - Y is the second numpy.ndarray of shape (m, ny) to shuffle,
    m is the same number of data points as in X and ny is the number
    of features in Y.
    - Returns: the shuffled X and Y matrices.
    """
    shuff = np.random.permutation(X.shape[0])
    X_s = X[shuff]
    Y_s = Y[shuff]
    return X_s, Y_s


def calculate_accuracy(y, y_pred):
    """
    - Calculates the accuracy of a prediction.
    - y is a placeholder for the labels of the input data.
    - y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction.
    """
    equality = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """
    - Calculates the softmax cross-entropy loss of a prediction.
    - y is a placeholder for the labels of the input data.
    - y_pred is a tensor containing the network’s predictions.
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    - Creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm.
    - loss is the loss of the network.
    - alpha is the learning rate.
    - beta1 is the weight used for the first moment.
    - beta2 is the weight used for the second moment.
    - epsilon is a small number to avoid division by zero.
    Returns: the Adam optimization operation.
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    train = optimizer.minimize(loss)
    return train


def create_batch_norm_layer(prev, n, activation, epsilon=1e-8):
    """
    - Creates a batch normalization layer for a neural network in tensorflow.
    - prev is the activated output of the previous layer.
    - n is the number of nodes in the layer to be created.
    - activation is the activation function that should
    be used on the output of the layer.
    - you should use an epsilon of 1e-8.
    Returns: a tensor of the activated output for the layer
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=w)
    z = model(prev)

    gamma = tf.Variable(tf.ones((z.get_shape()[-1])))
    beta = tf.Variable(tf.zeros((z.get_shape()[-1])))

    m, v = tf.nn.moments(z, [0])
    z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, epsilon)

    ypred = activation(z_norm)
    return ypred


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    - Creates a learning rate decay operation in tensorflow using
    inverse time decay.
    - alpha is the original learning rate.
    - decay_rate is the weight used to determine the rate at which
    alpha will decay.
    - global_step is the number of passes of gradient descent that
    have elapsed.
    - decay_step is the number of passes of gradient descent that
    should occur before alpha is decayed further.
    """
    # https://docs.w3cub.com/tensorflow~python/tf/train/inverse_time_decay
    decay = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return decay


def get_batch(t, batch_size):
    """
    Helper function to divide data in batches
    """

    batch_list = []
    i = 0
    m = t.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(t[i:(i + batch_size)])
        else:
            batch_list.append(t[i:])
        i += batch_size

    return batch_list


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    - Builds, trains, and saves a neural network model in
    tensorflow using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization.
    - Data_train is a tuple containing the training inputs and
    training labels, respectively.
    - Data_valid is a tuple containing the validation inputs and
    validation labels, respectively.
    - layers is a list containing the number of nodes in each layer
    of the network.
    - activation is a list containing the activation functions
    used for each layer of the network.
    - alpha is the learning rate.
    - beta1 is the weight for the first moment of Adam Optimization.
    - beta2 is the weight for the second moment of Adam Optimization.
    - epsilon is a small number used to avoid division by zero.
    - decay_rate is the decay rate for inverse time decay of the
    learning rate (the corresponding decay step should be 1.
    - batch_size is the number of data points that should be in a mini-batch.
    - epochs is the number of times the training should pass through the
    whole dataset.
    - save_path is the path where the model should be saved to.
    Returns: the path where the model was saved.
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    m = Data_train[0].shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign_add(global_step, 1,
                                          name='increment_global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs + 1):
            x_t, y_t = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run((loss, accuracy),
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if e < epochs:
                X_batch_t = get_batch(x_t, batch_size)
                Y_batch_t = get_batch(y_t, batch_size)
                for b in range(1, len(X_batch_t) + 1):
                    sess.run((increment_global_step, train_op),
                             feed_dict={x: X_batch_t[b - 1],
                             y: Y_batch_t[b - 1]})
                    loss_t, acc_t = sess.run((loss, accuracy),
                                             feed_dict={x: X_batch_t[b - 1],
                                                        y: Y_batch_t[b - 1]})
                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_t))
                        print('\t\tAccuracy: {}'.format(acc_t))
        save_path = saver.save(sess, save_path)
    return save_path
