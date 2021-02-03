#!/usr/bin/env python3
"""
performs forward propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    - Performs forward propagation over a convolutional layer of a
    neural network.
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer, m is the number of examples,
    h_prev is the height of the previous layer,
    w_prev is the width of the previous layer and c_prev is the number
    of channels in the previous layer.
    - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution, kh is the filter height,
    kw is the filter width, c_prev is the number of channels in
    the previous layer and c_new is the number of channels in the output.
    - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution.
    - activation is an activation function applied to the convolution.
    - padding is a string that is either same or valid, indicating the
    type of padding used.
    - stride is a tuple of (sh, sw) containing the strides for the convolution,
    sh is the stride for the height, sw is the stride for the width.
    Returns: the output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(((h_prev - 1)*sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1)*sw + kw - w_prev) / 2) + 1

    if padding == "valid":
        ph = 0
        pw = 0

    # Add zero padding to the input image
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    zh = int(((h_prev + (2*ph) - kh) / sh) + 1)
    zw = int(((w_prev + (2*pw) - kw) / sw) + 1)
    Z = np.zeros((m, zh, zw, c_new))
    for y in range(zh):
        for x in range(zw):
            for k in range(c_new):
                vert_start = y * sh
                vert_end = (y * sh) + kh
                horiz_start = x * sw
                horiz_end = (x * sw) + kw

                a_slice_prev = A_padded[:, vert_start:vert_end,
                                        horiz_start:horiz_end, :]
                # Element-wise product between a_slice and W.
                # Do not add the bias yet.
                prev_s = np.multiply(a_slice_prev, W[:, :, :, k])
                # Sum over all entries of the volume prev_s.
                sum_z = np.sum(prev_s, axis=(1, 2, 3))
                z1 = sum_z + b[:, :, :, k]

                Z[:, y, x, k] = activation(z1)
    return Z
