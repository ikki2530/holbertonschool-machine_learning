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
        """
        # W (1, nx), X (nx, m), b (1, 1)
        z = self.__W @ X + self.__b
        self.__A = 1/ (1 + np.exp( -1 * z))
        return self.__A
