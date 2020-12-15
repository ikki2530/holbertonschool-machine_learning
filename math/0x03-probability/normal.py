#!/usr/bin/env python3
"""Normal distribution"""


class Normal():

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data: is a list of the data to be used to estimate the distribution
        mean: is the mean of the distribution
        stddev: is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            std = 0
            for i in range(len(data)):
                std += (data[i] - self.mean) ** 2
            self.stddev = std / len(data)
