#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial():
    """Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        """
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p => 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                TypeError("data must be a list")
            if len(data) < 2:
                ValueError("data must contain multiple values")
            mean = sum(data) / len(data)  # mean = n*p
            std = 0
            for i in range(len(data)):
                std += (data[i] - mean) ** 2
            var = (std / len(data))
            self.p = float(1 - (var / mean))
            n = mean / self.p
            self.n = int(round(n))
            # after round n, must recalculate new p
            self.p = mean / self.n
