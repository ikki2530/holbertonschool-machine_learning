#!/usr/bin/env python3
"""Initialize Poisson"""
from math import exp


class Poisson():
    """Poisson class"""

    def __init__(self, data=None, lambtha=1.):
        """
        Poisson distribution constructor
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
            # lambtha
            c = len(data)
            suma = 0
            for i in range(c):
                suma += data[i]
            self.lambtha = suma / c

    def pmf(self, k):
        """
        pmf instance method
        k: is the number of “successes”
        returns: CDF value for k
        """
        if type(k) not in (int, float):
            k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(1, k + 1):
            fact = i * fact
        return (self.lambtha ** k) * exp(- self.lambtha) / fact
