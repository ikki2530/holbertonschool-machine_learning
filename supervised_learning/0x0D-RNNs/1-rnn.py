#!/usr/bin/env python3
"""performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i).
        * t is the maximum number of time steps.
        * m is the batch size.
        * i is the dimensionality of the data.
    - h_0 is the initial hidden state, given as a numpy.ndarray
    of shape (m, h).
        * h is the dimensionality of the hidden state.
    - Returns: H, Y.
        * H is a numpy.ndarray containing all of the hidden states.
        * Y is a numpy.ndarray containing all of the outputs.
    """
    T_x, m, i = X.shape
    h, o = rnn_cell.Wy.shape
    # print("T_x", T_x, " | m", m, "| i", i, " | o", o, " | h", h)
    H = np.zeros((T_x + 1, m, h))  # h is n_a

    Y = np.zeros((T_x, m, o))  # o is n_y
    H[0, :, :] = h_0
    for t in range(T_x):
        xt = X[t, :, :]
        h_next, yt_pred = rnn_cell.forward(H[t, :, :], xt)
        # print("h_next", h_next.shape)
        # print("H shape", H.shape)
        # print("yt_pred", yt_pred.shape)
        # print("Y shape", Y.shape)
        H[t + 1, :, :] = h_next
        Y[t, :, :] = yt_pred
    return H, Y
