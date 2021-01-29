#!/usr/bin/env python3
"""
performs a convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    - Performs a convolution on images using multiple kernels.
    - images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images, m is the number of images, h is the
    height in pixels of the images, w is the width in
    pixels of the images and w is the width in pixels of the images.
    - kernels is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernels for the convolution, kh is the height of a kernel,
    kw is the width of a kernel, nc is the number of kernels.
    - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’,
    ph is the padding for the height of the image and
    pw is the padding for the width of the image.
    - stride is a tuple of (sh, sw),
    sh is the stride for the height of the image
    and sw is the stride for the width of the image.
    Returns: a numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == "same":
        ph = int(((h - 1)*sh + kh - h) / 2) + 1
        pw = int(((w - 1)*sw + kw - w) / 2) + 1
    if padding == "valid":
        ph = 0
        pw = 0
    if type(padding) == tuple:
        ph, pw = padding

    # Add zero padding to the input image
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    # new dimensions
    zh = int(((h + (2*ph) - kh) / sh) + 1)
    zw = int(((w + (2*pw) - kw) / sw) + 1)
    output = np.zeros((m, zh, zw, nc))

    for y in range(zh):
        for x in range(zw):
            for k in range(nc):
                output[:, y, x, k] = np.sum(kernels[:, :, :, k] * image_padded[
                    :, y*sh:y*sh + kh, x*sw:x*sw + kw, :], axis=(1, 2, 3))
    return output
