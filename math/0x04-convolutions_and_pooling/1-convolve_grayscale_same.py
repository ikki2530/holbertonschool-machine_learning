#!/usr/bin/env python3
"""
performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    - Performs a same convolution on grayscale images.
    - images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images, m is the number of images,
    h is the height in pixels of the images and w is the width
    in pixels of the images.
    - kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution, kh is the height of the kernel
    and kw is the width of the kernel.
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    height = h
    width = w
    ph = max((kh - 1) // 2,
             kh // 2)
    pw = max((kw - 1) // 2,
             kw // 2)
    output = np.zeros((m, height, width))

    # Add zero padding to the input image
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))

    for y in range(height):
        for x in range(width):
            output[:, y, x] = np.sum(kernel * image_padded[
                   :, y:y + kh, x:x + kw], axis=(1, 2))

    return output
