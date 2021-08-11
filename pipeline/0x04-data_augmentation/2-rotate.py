#!/usr/bin/env python3
"""
Function that rotates an image by 90 degrees counter-clockwise
"""
import tensorflow as tf


def rotate_image(image):
    """
    Function that rotates an image by 90 degrees counter-clockwise
    Arguments:
     - image is a 3D tf.Tensor containing the image to rotate
    Returns:
     The rotated image
    """
    rotate = tf.image.rot90(image, k=1)
    return rotate
