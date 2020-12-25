#!/usr/bin/env python3
"""Neuron class"""
import numpy as np


class Neuron():
    """Neuron private class Neuron"""

    def __init__(self, nx):
        """
        Initialize a privatize Neuron class object
        nx: is the number of input features to the neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=nx).reshape((1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Get the W
        Returns:
            list of lists of weigths
        """
        return self.__W

    @property
    def b(self):
        """ Get the b
        Returns:
            bias
        """
        return self.__b

    @property
    def A(self):
        """ Get the A activation function
        Returns:
        """
        return self.__A

    def forward_prop(self, X):
        """
        defines a single neuron performing binary classification.
        X: is a numpy.ndarray with shape (nx, m) that contains the input data.
        Returns: A matrix with the values activated by sigmoid function (1, m)
        """
        # W (1, nx), X (nx, m), b (1, 1)
        z = self.__W @ X + self.__b
        self.__A = 1 / (1 + np.exp(-1 * z))
        return self.__A

    def cost(self, Y, A):
        """
        - Calculates the cost of the model using logistic regression.
        - Y is a numpy.ndarray with shape (1, m) that contains the
        correct labels for the input data.
        - A is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example.
        Returns the cost.
        """
        cost = -1 * np.sum(((Y * np.log(A)) + ((1 - Y) * np.log(
                1.0000001 - A))))
        cost = cost / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        - Evaluates the neuron’s predictions.
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        where nx is the number of input features to the neuron and m is the
        number of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)  # if A >= 1 then 1, 0 otherwise
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        where nx is the number of input features to the neuron and m is the
        number of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the
        correct labels for the input data.
        - A is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example.
        - alpha is the learning rate
        """
        m = X.shape[1]
        dz = A - Y
        dw = (dz @ A.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        where nx is the number of input features to the neuron and m is the
        number of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - iterations is the number of iterations to train over.
        - alpha is the learning rate.
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0.0:
            raise ValueError("alpha must be positive")
        for epoch in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        A, c = self.evaluate(X, Y)
        return A, c
