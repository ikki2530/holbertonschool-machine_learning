
import numpy as np


class RNNCell():
    def __init__(self, i, h, o):
        """
        Constructor.
        - i is the dimensionality of the data.
        - h is the dimensionality of the hidden state.
        - o is the dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

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
        h_next = np.concatenate((h_prev, x_t), axis=1)  # mx(h+i)
        h_next = np.tanh((h_next @ self.Wh) + self.bh)  # (mxhi)@(hi,h) = mxh
        y = (h_next @ self.Wy) + self.by   # (mxh)x(h, o) = mxo
        y = self.softmax(y)
        return h_next, y
