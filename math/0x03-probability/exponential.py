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

            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        x: is the time period
        Returns: PDF of value x
        """
        if x < 0:
            return 0
        e = 2.7182818284590452353602874
        exp_pdf = self.lambtha * (e ** (-1 * self.lambtha * x))
        return exp_pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        x: is the time period
        Returns: the CDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818284590452353602874
        suma_cdf = 1 - e ** (-x * self.lambtha)
        return suma_cdf
