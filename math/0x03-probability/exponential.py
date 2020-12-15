#!/usr/bin/env python3
"""Exponential distribution"""


class Exponential():
    """Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential class
        data: is a list of the data to be used to estimate the distribution
        lambtha: is the expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)
