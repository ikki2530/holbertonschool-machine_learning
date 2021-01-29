#!/usr/bin/env python3
"""
performs a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.
    - images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images, m is the number of images, h is the height
    in pixels of the images,
    w is the width in pixels of the images and c is the number
    of channels in the image.
    - kernel is a numpy.ndarray with shape (kh, kw, c)
    containing the kernel for the convolution, kh
    is the height of the kernel and
    kw is the width of the kernel.
    - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’,
    ph is the padding for the height of the image and pw is the
    padding for the width of the image.
    - stride is a tuple of (sh, sw), sh is the stride for
    the height of the image and sw is the stride for the width of the image.
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
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
    output = np.zeros((m, zh, zw))

    for y in range(zh):
        for x in range(zw):
            output[:, y, x] = np.sum(kernel * image_padded[
                   :, y*sh:y*sh + kh, x*sw:x*sw + kw, :], axis=(1, 2, 3))
    return output
