#!/usr/bin/env python3
"""
represents a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal():
    """
        Represents a Multivariate Normal distribution.
    """
    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set.
            - n is the number of data points
            - d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        _, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.
        - x is a numpy.ndarray of shape (d, 1) containing the
        data point whose PDF should be calculated.
            - d is the number of dimensions of the Multinomial instance.
        Returns the value of the PDF.
        """
        d, _ = self.mean.shape
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        dx, dy = x.shape
        if dx != d or dy != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        # mean (d, 1), cov (d, d), x (d, 1)
        # (1, d)
        cinv = np.linalg.inv(self.cov)
        exp = -0.5 * ((x - self.mean).T @ cinv @ (x - self.mean))
        den = ((2 * np.pi) ** d) * (np.linalg.det(self.cov))
        den = np.sqrt(den)

        dist = (1 / den) * np.exp(exp[0][0])

        return dist
