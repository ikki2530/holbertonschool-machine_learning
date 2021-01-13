#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """
    - data is the list of data to calculate the moving average of.
    - beta is the weight used for the moving average.
    Your moving average calculation should use bias correction.
    Returns: a list containing the moving averages of data.
    """
    v1 = 0
    mv_avrg = []
    for i in range(0, len(data)):
        v1 = beta * v1 + (1 - beta) * data[i]  # exponetially weighted average
        v2 = v1/(1 - beta ** (i + 1))
        mv_avrg.append(v2)
    return mv_avrg
