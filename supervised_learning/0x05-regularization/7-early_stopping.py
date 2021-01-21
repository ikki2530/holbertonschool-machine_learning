#!/usr/bin/env python3
"""
determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.
    - cost is the current validation cost of the neural network.
    - opt_cost is the lowest recorded validation cost of the neural network.
    - threshold is the threshold used for early stopping.
    - patience is the patience count used for early stopping.
    - count is the count of how long the threshold has not been met.
    Returns: a boolean of whether the network should be stopped early,
    followed by the updated count.
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if (count == patience):
        boolean = True
    else:
        boolean = False

    return boolean, count
