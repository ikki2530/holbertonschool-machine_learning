#!/usr/bin/env python3
"""
performs a convolution on grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    - images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images, m is the number of images,
    h is the height in pixels of the images and w is the width
    in pixels of the images.
    - kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution, kh is the
    height of the kernel and
    kw is the width of the kernel.
    - padding is a tuple of (ph, pw), ph is the padding
    for the height of the image
    and pw is the padding for the width of the image.
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    # Add zero padding to the input image
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))

    zh = h + (2*ph) - kh + 1
    zw = w + (2*pw) - kw + 1
    output = np.zeros((m, zh, zw))

    for y in range(zh):
        for x in range(zw):
            output[:, y, x] = np.sum(kernel * image_padded[
                   :, y:y + kh, x:x + kw], axis=(1, 2))

    return output
