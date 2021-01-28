#!/usr/bin/env python3
"""
performs a same convolution on grayscale images
"""
import numpy as np
from math import ceil, floor


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
    height = int(ceil(float(h)))
    width = int(ceil(float(w)))

    pad_along_height = max((height - 1) + kh - h, 0)
    pad_along_width = max((width - 1) + kw - w, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    output = np.zeros((m, height, width))

    # Add zero padding to the input image
    image_padded = np.zeros((m, h + pad_along_height, w + pad_along_width))
    image_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images

    for y in range(height):
        for x in range(width):
            output[:, y, x] = np.sum(kernel * image_padded[
                   :, y:y + kh, x:x + kw], axis=(1, 2))

    return output
