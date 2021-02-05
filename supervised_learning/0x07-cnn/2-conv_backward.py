#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.
    - performs back propagation over a convolutional layer of a neural network.
    - dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated
    output of the convolutional layer.
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer, h_prev is the height
    of the previous layer, w_prev is the width of the previous layer,
    c_prev is the number of channels in the previous layer.
    - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution, kh is the filter height,
    kw is the filter width.
    - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing
    the biases applied to the convolution.
    - padding is a string that is either same or valid,
    indicating the type of padding used.
    - stride is a tuple of (sh, sw) containing
    the strides for the convolution, sh is the stride for the height
    and sw is the stride for the width.
    Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    # select padding
    if padding == "same":
        ph = int(((h_prev - 1)*sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1)*sw + kw - w_prev) / 2) + 1

    if padding == "valid":
        ph = 0
        pw = 0

    # Retrieve dimensions from dZ's shape
    m, h_new, w_new, c_new = dZ.shape

    A_pad = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant')
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    for i in range(m):
        for y in range(h_new):
            for x in range(w_new):
                z = y * sh
                w = x * sw
                for k in range(c_new):
                    aux1 = dZ[i, y, x, k]
                    aux2 = A_pad[i, z: z + kh, w: w + kw, :]
                    dA[i, z: z + kh, w: w + kw, :] += aux1 * W[:, :, :, k]
                    dW[:, :, :, k] += aux1 * aux2
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
