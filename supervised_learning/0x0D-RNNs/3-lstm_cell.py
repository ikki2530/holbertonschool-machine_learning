#!/usr/bin/env python3
"""implement a class LSTMCell that represents an LSTM unit"""
import numpy as np


class LSTMCell():
    """implement a class LSTMCell that represents an LSTM unit"""
    def __init__(self, i, h, o):
        """
        Constructor
        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.
        - Creates the public instance attributes
        Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by that
        represent the weights and biases of the cell.
            * Wf and bf are for the forget gate
            * Wu and bu are for the update gate
            * Wc and bc are for the intermediate cell state
            * Wo and bo are for the output gate
            * Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        - performs forward propagation for one time step.
        - x_t is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell.
        - h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state.
        - c_prev is a numpy.ndarray of shape (m, h)
        containing the previous cell state.
        - Returns: h_next, c_next, y
            * h_next is the next hidden state
            * c_next is the next cell state
            * y is the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid((concat @ self.Wf) + self.bf)
        ut = self.sigmoid((concat @ self.Wu) + self.bu)
        cct = np.tanh((concat @ self.Wc) + self.bc)
        c_next = (ut * cct) + (ft * c_prev)
        ot = self.sigmoid((concat @ self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)

        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
