#!/usr/bin/env python3
"""implement a class GRUCell that represents a gated recurrent unit"""
import numpy as np


class GRUCell():
    """class GRUCell that represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """
        Constructor
        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.
        - Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell.
            * Wzand bz are for the update gate.
            * Wrand br are for the reset gate.
            * Whand bh are for the intermediate hidden state.
            * Wyand by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    # Function
    def sigmoid(self, x):
        """
        Sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(-x))

        return sigmoid

    def forward(self, h_prev, x_t):
        """
        - performs forward propagation for one time step.
        - x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell.
            * m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state.
        - The output of the cell should use a softmax activation function.
        Returns: h_next, y
            * h_next is the next hidden state.
            * y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        fr = (concat @ self.Wr) + self.br
        fr = self.sigmoid(fr)
        fz = (concat @ self.Wz) + self.bz
        fz = self.sigmoid(fz)
        fr_c = fr * h_prev
        concat2 = np.concatenate((fr_c, x_t), axis=1)
        cct = np.tanh((concat2 @ self.Wh) + self.bh)
        h_next = (fz * cct) + ((1 - fz) * h_prev)
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, y
