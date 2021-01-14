#!/usr/bin/env python3
"""updates a variable in place using the Adam optimization algorithm"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    - Updates a variable in place using the Adam optimization algorithm.
    - alpha is the learning rate.
    - beta1 is the weight used for the first moment.
    - beta2 is the weight used for the second moment.
    - epsilon is a small number to avoid division by zero.
    - var is a numpy.ndarray containing the variable to be updated.
    - grad is a numpy.ndarray containing the gradient of var.
    - v is the previous first moment of var.
    - s is the previous second moment of var.
    - t is the time step used for bias correction.
    Returns: the updated variable, the new first moment,
    and the new second moment, respectively.
    """
    # momentum
    v1 = (beta1 * v) + (1 - beta1) * grad
    # RMSProp
    v2 = (beta2 * s) + (1 - beta2) * (grad ** 2)
    # var1 = var - ((alpha * grad) / ((v2**0.5) + epsilon))
    v1corr = v1 / (1 - (beta1**t))
    v2corr = v2 / (1 - (beta2**t))

    # update
    var = var - ((alpha * v1corr) / ((v2corr ** 0.5) + epsilon))

    return var, v1, v2
