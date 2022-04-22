#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:49:48 2021
# Numerical proof-of-concept of Proposition 3 in the paper,
# according to which multivariate Pareto type III random variables
# can be generated as gamma-weighted mixtures of independent Weibull random variables.
# Draws a sample, plots its histogram, and compares the sample estimate of the moments with their theoretical predictions.
@author: Stefan Bucher (web@stefan-bucher.ch)
"""

import numpy as np
import pandas as pd
from scipy.special import gamma
import matplotlib.pyplot as plt

nDraws =  10000000 # sample size (large in order to ensure convergence to theoretical values)
beta = 3 # shape parameter
mu = 0*np.array([1,1]) # vector of location parameters mu. also determines dimensionality of the distribution
sigma = 1*np.array([1,1]) # vector of scale parameters sigma


##############################################
# Drawing from the Mixture Model
#############################################

# Draw samples using the gamma-weighted mixture of independent Weibull random variables (Proposition 3)
def randomParetoSample_mixtureModel(beta, mu, sigma, size=1):
    n = len(mu)
    S = np.empty([nDraws, n])  # nSamples x n. Each row contains one random sample of the distribution.
    for draw in range(size):
        Z = np.random.gamma(shape=1.0, scale=1.0, size=1) # gamma-weighted weighting
        U = np.random.exponential(scale=1.0, size=n) # vector of independent exponential random variables
        for i in range(n):
            S[draw,i] = mu[i] + sigma[i] * (U[i] / Z)**(1/beta) # S is a mixture of Weibull (transformed exponential) r.v.s
    return S


##############################################
# Histogram of a Random Sample
##############################################

S = randomParetoSample_mixtureModel(beta, mu, sigma, size=nDraws)

# Joint histogram
plt.figure(figsize=(10,10))
plt.hist2d(S[:,0],S[:,1], bins=100, range=[[0, 5], [0, 5]])
plt.show()

# (Columnwise normalized) conditional histogram ('bow-tie plot')
hist = pd.DataFrame(plt.hist2d(S[:,0],S[:,1], bins=100, range=[[0, 5], [0, 5]], density=True)[0])
conditionalHist = hist.divide(hist.sum(axis=0), axis=1)
conditionalHist = conditionalHist.divide(conditionalHist.max(axis=0)-conditionalHist.min(axis=0), axis=1)
plt.figure(figsize=(10,10))
plt.imshow(conditionalHist)
plt.gca().invert_yaxis()
plt.show()

##############################################
# COMPARISON WITH THEORETICAL FORMULAE
##############################################

# Variance formula (Appendix - Derivation of Moments)
def theoretical_variance(beta, sigma):
    return sigma**2 * ( gamma(1-2/beta)*gamma(1+2/beta) - ( gamma(1-1/beta)*gamma(1+1/beta) )**2 )

# Equivalent variance expression using trigonometric expressions (Appendix - Derivation of Moments)
def theoretical_variance_trigonometric(beta, sigma):
    return sigma**2 * ( (2*np.pi/beta)/np.sin(2*np.pi/beta) - ((np.pi/beta)/np.sin(np.pi/beta))**2 )

# Covariance formula (eq. 9); only applies to off-diagonal elements
def theoretical_covariance(beta, sigma):
    tmp = (gamma(1 + 1 / beta)) ** 2 * (gamma(1 - 2 / beta) - (gamma(1 - 1 / beta)) ** 2)
    return np.array([[None, tmp * sigma[0] * sigma[1]],
                           [tmp * sigma[1] * sigma[0], None]])

# Correlation formula (Appendix - Derivation of Moments); only applies to off-diagonal elements
def theoretical_correlation(beta):
    tmp = ( gamma(1-2/beta) - (gamma(1-1/beta))**2 ) / ( (gamma(1+2/beta)/(gamma(1+1/beta))**2) * gamma(1-2/beta) - (gamma(1-1/beta))**2 )
    return np.array([[1,tmp],
                     [tmp,1]])



print("THEORETICAL COVARIANCE MATRIX:")
print(theoretical_covariance(beta,sigma))

print("EMPIRICAL COVARIANCE MATRIX:")
print(np.cov(S, rowvar=False))

print("THEORETICAL VARIANCE:")
print(theoretical_variance(beta,sigma))

print("THEORETICAL VARIANCE (trigonometric):")
print(theoretical_variance_trigonometric(beta,sigma))

print("EMPIRICAL VARIANCE:")
print(np.var(S, axis=0))

print("THEORETICAL CORRELATION MATRIX:")
print(theoretical_correlation(beta))

print("EMPIRICAL CORRELATION MATRIX:")
print(np.corrcoef(S, rowvar=False))

