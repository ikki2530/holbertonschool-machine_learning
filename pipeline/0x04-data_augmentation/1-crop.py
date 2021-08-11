#!/usr/bin/env python3
"""
Crop function
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Function that performs a random crop of an image
    """
    croped = tf.random_crop(image, size)
    return croped
