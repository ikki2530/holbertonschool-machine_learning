#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class NeuralNetwork():
    """
    Neural Network
    """

    def __init__(self, nx, nodes):
        """
        - Initialize a neural network
        - nx is the number of input features
        - nodes is the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Get the W1 weight
        Returns:
            list of lists of weigths
        """
        return self.__W1

    @property
    def b1(self):
        """ Get the b1 bias
        Returns:
            bias
        """
        return self.__b1

    @property
    def A1(self):
        """ Get the A1 activation function
        Returns:
        """
        return self.__A1

    @property
    def W2(self):
        """ Get the W2 weight
        Returns:
            list of lists of weigths
        """
        return self.__W2

    @property
    def b2(self):
        """ Get the b2 bias
        Returns:
            bias
        """
        return self.__b2

    @property
    def A2(self):
        """ Get the A2 activation function
        Returns:
        """
        return self.__A2
