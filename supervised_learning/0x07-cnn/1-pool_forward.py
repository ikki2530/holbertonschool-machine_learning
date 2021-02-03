#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
    https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
    - Performs forward propagation over a pooling layer of a neural network.
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer,
    m is the number of examples, h_prev is the height of the previous layer,
    w_prev is the width of the previous layer
    and c_prev is the number of channels in the previous layer.
    - kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling,kh is the kernel height and kw is the kernel width.
    - stride is a tuple of (sh, sw) containing the strides for the pooling,
    sh is the stride for the height and sw is the stride for the width.
    - mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively.
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # padding = 0
    zh = int(((h_prev - kh) / sh) + 1)
    zw = int(((w_prev - kw) / sw) + 1)

    A = np.zeros((m, zh, zw, c_prev))

    for y in range(zh):
        for x in range(zw):
            vert_start = y * sh
            vert_end = (y * sh) + kh
            horiz_start = x * sw
            horiz_end = (x * sw) + kw
            a_slice_prev = A_prev[:, vert_start:vert_end,
                                  horiz_start:horiz_end, :]
            if mode == "max":
                A[:, y, x, :] = np.max(a_slice_prev, axis=(1, 2))
            elif mode == "average":
                A[:, y, x, :] = np.average(a_slice_prev, axis=(1, 2))
    # Making sure your output shape is correct
    # assert(A.shape == (m, zh, zw, c_prev))
    return A
