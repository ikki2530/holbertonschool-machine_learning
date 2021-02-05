#!/usr/bin/env python3
"""
performs back propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    - Performs back propagation over a pooling layer of a neural network.
    - dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer,
    m is the number of examples, h_new is the height of the output,
    w_new is the width of the output, c is the number of channels.
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer, h_prev is the height
    of the previous layer, w_prev is the width of the previous layer.
    - kernel_shape is a tuple of (kh, kw) containing the size of
    the kernel for the pooling, kh is the kernel height and
    kw is the kernel width.
    - stride is a tuple of (sh, sw) containing the strides for the pooling,
    sh is the stride for the height and sw is the stride for the width.
    - mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively.
    Returns: the partial derivatives with respect to the previous
    layer (dA_prev).
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    sh, sw = stride
    kh, kw = kernel_shape

    daprev = np.zeros_like(A_prev)

    for n in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for k in range(c):
                    if mode == "max":
                        tmp = A_prev[
                            n, y * sh:y * sh + kh, x * sw:x * sw + kw, k
                        ]
                        mask = np.where(tmp == np.max(tmp), 1, 0)
                        daprev[n, y * sh:y * sh + kh, x * sw:x * sw + kw, k
                               ] += dA[n, y, x, k] * mask
                    else:
                        avg = dA[n, y, x, k] / (kw * kh)
                        daprev[
                            n, y * sh:y * sh + kh, x * sw:x * sw + kw, k
                        ] += np.ones(kernel_shape) * avg
    return daprev
