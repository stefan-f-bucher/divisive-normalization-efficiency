#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Stefan Bucher (web@stefan-bucher.ch), 2021
"""
import numpy as np
import scipy.special
import math
import string
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.widgets import Slider
import matplotlib.ticker as ticker

from pdfs import *
from mixtureModel import randomParetoSample_mixtureModel
from plotFunctions import *


nDraws =  100000 # sample size for histograms
gridResolution = 0.01 
lowGridResolution = 0.1
betas = [ 0.5, 0.75, 1, 2, 3, 5, 10, 20]

fileFormat = '.pdf' # png or jpg seem to give better results than pdf, eps, svg
cmap = 'binary' # colormap ('gray': white for high values; 'binary': black for high values)
vmin = 0 # ensure that white in 'binary' color map corresponds to pdf=0.
vmax = None # chosen automatically based on data

def trunc_joint(beta):
    return 3 if (beta > 1) else 1

def trunc_cond(beta):
    return 10 if (beta >= 1) else 3

def trunc_condhist(beta):
    return 10 if (beta < 5) else 3

##############################################
# Plotting Univariate Example
##############################################
x1d = np.arange(0,4.001,gridResolution)
y1d = np.arange(0,1.001,gridResolution)

# univariate Pareto pdf with mu=0 and sigma=1
def univPareto_pdf(x, beta):
    return beta * x**(beta-1) / (1+x**beta)**2

def univDivNorm(x, alpha, gamma=1, b=1, lbda=1):
    return gamma * x**alpha / (b**alpha + lbda * x**alpha)

plotUnivariateDensity(X=x1d, density=univPareto_pdf(x1d, beta=1), xlabel='$s$', ylabel='$f_S(s)$', fname='univariateExample_input_beta1'+fileFormat)
plotUnivariateDensity(X=x1d, density=univPareto_pdf(x1d, beta=2), xlabel='$s$', ylabel='$f_S(s)$', fname='univariateExample_input_beta2'+fileFormat)
plotUnivariateDensity(X=x1d, density=univDivNorm(x1d, alpha=1), xlabel='$x=s-\mu$', ylabel='$y=r(x)$', ylimlabel='$\gamma/\lambda$', fname='univariateExample_DN_alpha1'+fileFormat)
plotUnivariateDensity(X=x1d, density=univDivNorm(x1d, alpha=2), xlabel='$x=s-\mu$', ylabel='$y=r(x)$', ylimlabel='$\gamma/\lambda$', fname='univariateExample_DN_alpha2'+fileFormat)
plotUnivariateDensity(X=y1d, density=0.5*np.ones(y1d.size), xlabel='$y$', ylabel='$f_Y(y)$', xlimlabel='$\gamma/\lambda$', ylimlabel='$2\lambda/\gamma$', fname='univariateExample_output'+fileFormat)


##############################################
# Plotting Histogram of a Random Sample
##############################################
for beta in betas:
    S = randomParetoSample_mixtureModel(beta=beta, mu=0*np.array([1,1]), sigma=1*np.array([1,1]), size=nDraws)

    plotJointHistogram(S, trunc=trunc_joint(beta), xlabel='$s_1$', ylabel='$s_2$', x0label='$\mu_1$', y0label='$\mu_2$', cmap=cmap, fname='jointHistogram_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

    plotConditionalHistogram(S, trunc=trunc_condhist(beta), xlabel='$s_1$', ylabel='$s_2$', x0label='$\mu_1$', y0label='$\mu_2$', fname='conditionalHistogram_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

    # Full Bowtie Histogram
    S_bowtie = np.vstack((np.array([1, 1])*randomParetoSample_mixtureModel(beta=beta, mu=0*np.array([1,1]), sigma=1*np.array([1,1]), size=int(nDraws)),
                          np.array([1, -1]) * randomParetoSample_mixtureModel(beta=beta, mu=0 * np.array([1, 1]), sigma=1 * np.array([1, 1]), size=int(nDraws)),
                          np.array([-1, 1]) * randomParetoSample_mixtureModel(beta=beta, mu=0 * np.array([1, 1]), sigma=1 * np.array([1, 1]), size=int(nDraws)),
                          np.array([-1, -1]) * randomParetoSample_mixtureModel(beta=beta, mu=0 * np.array([1, 1]), sigma=1 * np.array([1, 1]), size=int(nDraws))
                          ))
    plotConditionalHistogram(S_bowtie, trunc=trunc_condhist(beta), xlabel='$s_1$', ylabel='$s_2$', fullBowtie=True, fname='conditionalHistogram_fullBowtie_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

##############################################
# Mesh grids of different size for different purposes
##############################################
x1_full, x2_full = np.meshgrid(np.arange(0,100,gridResolution),np.arange(0,100,gridResolution))
x1_divnorm, x2_divnorm = np.meshgrid(np.arange(0,3.01,gridResolution),np.arange(0,3.01,gridResolution))
y1, y2 = np.meshgrid(np.arange(0,1.01,gridResolution),np.arange(0,1.01,gridResolution))
x1_fullbowtie, x2_fullbowtie = np.meshgrid(np.arange(-100,100,lowGridResolution),np.arange(-100,100,lowGridResolution))


#########################################
# Plotting Input Distributions
#########################################
for beta in betas:
    if beta<=1:
        zlim = 3
    else:
        zlim = 1

    x1_trunc, x2_trunc = np.meshgrid(np.arange(0, trunc_joint(beta) + 0.01, gridResolution), np.arange(0, trunc_joint(beta) + 0.01, gridResolution))

    ## Pareto distribution
    plotJointDensity(x1_trunc, x2_trunc, bivariatePareto_pdf(x1_full, x2_full, sigma1=1, sigma2=1, beta=beta), xlabel='$s_1$', ylabel='$s_2$', x0label='$\mu_1$', y0label='$\mu_2$',cmap=cmap, vmin=vmin, vmax=vmax, fname='Pareto_pdf_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

    plot3D(x2_trunc, x1_trunc, bivariatePareto_pdf(x2_full, x1_full, sigma1=1, sigma2=1, beta=beta), 
           xlabel='$s_2$', ylabel='$s_1$', x0label='$\mu_2$', y0label='$\mu_1$',  zlabel=None, fname='3D_Pareto_pdf_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat, show_pdf=False, contourProjections=True, zlim=zlim)
    
    ## Input under constant costs (just as a sanity check - should be identical to Pareto_pdf)
    plotJointDensity(x1_trunc, x2_trunc, input_pdf_from_output_pdf(x1_full, x2_full, gamma=1, b=1, beta=beta, lambda1=1, lambda2=1, output_pdf=bivariateUniform_pdf), 
           xlabel='$s_1$', ylabel='$s_2$', x0label='$\mu_1$', y0label='$\mu_2$',cmap=cmap, vmin=vmin, vmax=vmax, fname='input_pdf_constantCost_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

    plot3D(x2_trunc, x1_trunc, input_pdf_from_output_pdf(x2_full, x1_full, gamma=1, b=1, beta=beta, lambda1=1, lambda2=1, output_pdf=bivariateUniform_pdf), 
           xlabel='$s_2$', ylabel='$s_1$', x0label='$\mu_2$', y0label='$\mu_1$',  zlabel=None, fname='3D_input_pdf_constantCost_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat, show_pdf=False, contourProjections=True, zlim=zlim)
    
    ## Input under linear costs
    plotJointDensity(x1_trunc, x2_trunc, input_pdf_from_output_pdf(x1_full, x2_full, gamma=1, b=1, beta=beta, lambda1=1, lambda2=1, output_pdf=bivariateTruncatedExponential_pdf),
           xlabel='$s_1$', ylabel='$s_2$', x0label='$\mu_1$', y0label='$\mu_2$',cmap=cmap, vmin=vmin, vmax=vmax, fname='input_pdf_linearCost_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)
    
    plot3D(x2_trunc, x1_trunc, input_pdf_from_output_pdf(x2_full, x1_full, gamma=1, b=1, beta=beta, lambda1=1, lambda2=1, output_pdf=bivariateTruncatedExponential_pdf), 
           xlabel='$s_2$', ylabel='$s_1$', x0label='$\mu_2$', y0label='$\mu_1$',  zlabel=None, fname='3D_input_pdf_linearCost_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat, show_pdf=False, contourProjections=True, zlim=zlim)


#########################################
# Plotting Divisve Normalization Function
#########################################
for beta in betas:
    plot3D(x2_divnorm, x1_divnorm, r_i(x_i=x1_divnorm, x_j=x2_divnorm, gamma=1, b=1, beta=beta, lambda_i=1, lambda_j=1),
           xlabel='$x_j$', ylabel='$x_i$', zlabel='$r_i(x_i,x_j)$', fname='divNorm_fnct_alpha' + str(beta).translate(str.maketrans('', '', string.punctuation)) + fileFormat,
           contourProjections=False, zlim=1, zlimlabel='$\gamma$', elev=22, azim=13)


#########################################
# Plotting Output Distributions
#########################################

## Truncated Uniform
plotJointDensity(y1, y2, bivariateUniform_pdf(y1, y2, gamma=1), xlabel='$y_1$', ylabel='$y_2$', xlimlabel='$\gamma$', ylimlabel='$\gamma$',cmap=cmap, vmin=vmin, vmax=vmax, fname='truncatedUniform_pdf'+fileFormat)
    
plot3D(y2, y1, bivariateUniform_pdf(y2, y1, gamma=1),
       xlabel='$y_2$', ylabel='$y_1$',zlabel=None, fname='3D_truncatedUniform_pdf'+fileFormat, show_pdf=False, contourProjections=True, zlim=5, xlimlabel='$\gamma$', ylimlabel='$\gamma$')

## Truncated Exponential
plotJointDensity(y1, y2, bivariateTruncatedExponential_pdf(y1, y2, gamma=1), xlabel='$y_1$', ylabel='$y_2$', xlimlabel='$\gamma$', ylimlabel='$\gamma$',cmap=cmap, vmin=vmin, vmax=vmax, fname='truncatedExponential_pdf'+fileFormat)

plot3D(y2, y1, bivariateTruncatedExponential_pdf(y2,y1,gamma=1), 
       xlabel='$y_2$', ylabel='$y_1$',zlabel=None, fname='3D_truncatedExponential_pdf'+fileFormat, show_pdf=False, contourProjections=True, zlim=5, xlimlabel='$\gamma$', ylimlabel='$\gamma$')


#########################################
# Plotting Conditional Input Distribution ("Bow-tie plot")
#########################################

# top-right quadrant (positive x only)
for beta in betas:
    x1_cond, x2_cond = np.meshgrid(np.arange(0, trunc_cond(beta) + 0.01, gridResolution), np.arange(0, trunc_cond(beta) + 0.01, gridResolution))  # could do negative values for bow-tie plot but pdf is 0
    plotConditionalDensity(x1_cond, x2_cond, bivariatePareto_pdf(x1_full, x2_full, sigma1=1, sigma2=1, beta=beta), xlabel='$s_1$', ylabel='$s_2$', cmap='gray', fname='conditional_Pareto_pdf_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)

# full bow-tie plot (taking absolute value of x)
for beta in betas: #[1,2]:
    x1_bowtie, x2_bowtie = np.meshgrid(np.arange(-(trunc_cond(beta) + 0.11), trunc_cond(beta) + 0.11, lowGridResolution), np.arange(-(trunc_cond(beta) + 0.11), trunc_cond(beta) + 0.11, lowGridResolution))
    plotConditionalDensity(x1_bowtie, x2_bowtie, bivariatePareto_pdf(x1_fullbowtie, x2_fullbowtie, sigma1=1, sigma2=1, beta=beta, absoluteValue=True), xlabel='$s_1$', ylabel='$s_2$', cmap='gray', fname='conditional_Pareto_pdf_fullBowtie_beta'+str(beta).translate(str.maketrans('', '', string.punctuation))+fileFormat)
