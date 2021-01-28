#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """
    - Performs a valid convolution on grayscale images.
    - images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images, m is the number of images,
    h is the height in pixels of the images, w is the width in
    pixels of the images.
    - kernel is a numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution.
    kh is the height of the kernel and kw is the width of the kernel.
    Returns: a numpy.ndarray containing the convolved images
    # """
    m, h, w = images.shape
    kh, kw = kernel.shape
    height = int(ceil(float(h - kh + 1) / 1))
    width = int(ceil(float(w - kw + 1) / 1))

    output = np.zeros((m, height, width))
    for x in range(width):
        for y in range(height):
            output[:, y, x] = (kernel * images[
                               :, y*1:y*1 + kh, x*1:x*1 + kw]).sum()
    return output
