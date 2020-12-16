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
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                TypeError("data must be a list")
            if len(data) < 2:
                ValueError("data must contain multiple values")
            mean = sum(data) / len(data)  # mean = n*p
            var = sum([(data[x] - mean) ** 2 for x in range(len(data))]
                      ) / len(data)
            self.p = (1 - (var / mean))
            n = mean / self.p
            self.n = int(round(n))
            # after round n, must recalculate new p
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        Returns the PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        bin_pdf = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return bin_pdf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
        Returns the CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        cumulative = 0
        for i in range(k + 1):
            cumulative += self.pmf(i)
        return cumulative


def factorial(k):
    """calculates the factorial of k"""
    if k == 0:
        return 1
    fact = k * factorial(k - 1)
    return fact
