#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Generates multivariate Pareto type III random variables as gamma-weighted mixtures of independent Weibull random variables.
@author: Stefan Bucher (web@stefan-bucher.ch)
"""

import numpy as np
from scipy.special import gamma

# Draw samples using the gamma-weighted mixture of independent Weibull random variables (Proposition 3)
def randomParetoSample_mixtureModel(beta, mu, sigma, size=1):
    n = len(mu)
    S = np.empty([size, n])  # nSamples x n. Each row contains one random sample of the distribution.
    for draw in range(size):
        Z = np.random.gamma(shape=1.0, scale=1.0, size=1) # gamma-weighted weighting
        U = np.random.exponential(scale=1.0, size=n) # vector of independent exponential random variables
        for i in range(n):
            S[draw,i] = mu[i] + sigma[i] * (U[i] / Z)**(1/beta) # S is a mixture of Weibull (transformed exponential) r.v.s
    return S
