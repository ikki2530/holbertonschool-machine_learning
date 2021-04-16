#!/usr/bin/env python3
"""Class BayesianOptimization Bayesian
optimization on a noiseless 1D Gaussian process"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Constructor
        - f is the black-box function to be optimized.
        - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function.
        - Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init.
        - t is the number of initial samples.
        - bounds is a tuple of (min, max) representing the bounds of
        the space in which to look for the optimal point.
        - ac_samples is the number of samples that should be analyzed
        during acquisition.
        - l is the length parameter for the kernel.
        - sigma_f is the standard deviation given to the output of the
        black-box function.
        - xsi is the exploration-exploitation factor for acquisition.
        -minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.minimize = minimize
        self.xsi = xsi
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """Method that calculates the next best sample location
        - Returns: X_next, EI
            * X_next is a numpy.ndarray of shape (1,) representing the next
            best sample point.
            * EI is a numpy.ndarray of shape (ac_samples,) containing the
            expected improvement of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize:
                musopt = np.min(self.gp.Y)
                imp = (musopt - mu - self.xsi).reshape(-1, 1)
            else:
                musopt = np.amax(self.gp.Y)
                imp = (mu - musopt - self.xsi).reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return (X_next, ei.reshape(-1))

    def optimize(self, iterations=100):
        """Method that calculates the next best sample location"""
        for i in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if (X_next == self.gp.X).any():
                self.gp.X = self.gp.X[:-1]
                break
            self.gp.update(X_next, Y_next)
        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        Y_opt = self.gp.Y[index]
        X_opt = self.gp.X[index]
        return (X_opt, Y_opt)
