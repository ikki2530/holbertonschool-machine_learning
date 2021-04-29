#!/usr/bin/env python3
"""
Class RNNCell
"""
import numpy as np


class RNNCell():
    """Class RNNCell represents a cell of RNN"""
    def __init__(self, i, h, o):
        """
        Constructor.
        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Function to compute softmax values for each sets of scores in x
        """
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)

        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        - x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell.
            * m is the batche size for the data.
        - h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state.
        - Returns: h_next, y
            * h_next is the next hidden state.
            * y is the output of the cell.
        """
        h_next = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh((h_next @ self.Wh) + self.bh)
        y = (h_next @ self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y
