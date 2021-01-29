#!/usr/bin/env python3
"""
Performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    - images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images, m is the number of images,
    h is the height in pixels of the images,
    w is the width in pixels of the images,
    c is the number of channels in the image.
    - kernel_shape is a tuple of (kh, kw) containing
    the kernel shape for the pooling,
    kh is the height of the kernel and kw is the
    width of the kernel.
    - stride is a tuple of (sh, sw), sh is the stride for the height
    of the image and sw is the stride for the width of the image.
    - mode indicates the type of pooling, max indicates max pooling
    and avg indicates average pooling.
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    zh = int(((h - kh) / sh) + 1)
    zw = int(((w - kw) / sw) + 1)

    pooled = np.zeros((m, zh, zw, c))
    zm = np.arange(0, m)

    for i in range(zh):
        for j in range(zw):
            if mode == 'max':
                data = np.max(images[zm, i*sh:(
                              i*sh) + kh, j*sw:(j*sw) + kw], axis=(1, 2))
            if mode == 'avg':
                data = np.mean(images[zm, i*sh:(
                              i*sh) + kh, j*sw:(j*sw) + kw], axis=(1, 2))
            pooled[zm, i, j] = data
    return pooled
