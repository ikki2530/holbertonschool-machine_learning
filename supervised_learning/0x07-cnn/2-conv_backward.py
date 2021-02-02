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

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    for i in range(m):
        # select ith trainning example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for y in range(h_new):
            for x in range(w_new):
                for k in range(c_new):
                    vert_start = y
                    vert_end = vert_start + kh
                    horiz_start = x
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    # Update gradients for the window and the
                    # filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                :] += W[:, :, :, k] * dZ[i, y, x, k]
                    dW[:, :, :, k] += a_slice * dZ[i, y, x, k]
                    db[:, :, :, k] += dZ[i, y, x, k]
        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        dA_prev[i, :, :, :] = da_prev_pad

    # Making sure your output shape is correct
    # assert(dA_prev.shape == (m, h_prev, w_prev, c_prev))

    return dA_prev, dW, db
