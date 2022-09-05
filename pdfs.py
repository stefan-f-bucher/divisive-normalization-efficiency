#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file implements the pdfs of different input/output distributions for use in plotting.
@author: Stefan Bucher (web@stefan-bucher.ch)
"""
import numpy as np
import pandas as pd
import math

# all functions below take meshgrids as input and output

# Divisive Normalization Transformation; bivariate special case
def r_i(x_i, x_j, gamma, b, beta, lambda_i, lambda_j):
    return gamma * x_i**beta / (b**beta + lambda_i * x_i**beta + lambda_j * x_j**beta)


####################################
# Input PDFs
####################################

# pdf of bivariate Pareto distribution with mu=0
def bivariatePareto_pdf(x1, x2, sigma1, sigma2, beta, absoluteValue=False):
    n = 2
    if absoluteValue:
        x1 = np.abs(x1)
        x2 = np.abs(x2)
    with np.errstate(divide='ignore'):
        density = beta**n * math.factorial(n) *  (1/sigma1) * (x1/sigma1)**(beta-1) * (1/sigma2) * (x2/sigma2)**(beta-1) / ((1 + (x1/sigma1)**beta + (x2/sigma2)**beta )**(n+1))
        density = np.where(np.isinf(density), np.nan, density)
    # Sanity Check: This pdf already integrates to 1 so no normalization needed
    #domain_x, domain_y = np.meshgrid(np.linspace(0,100,10000), np.linspace(0,100,10000))  
    #dx = (np.max(domain_x)-np.min(domain_x))/(np.size(domain_x,1)-1)
    #dy = (np.max(domain_y)-np.min(domain_y))/(np.size(domain_y,0)-1)
    #density_tmp = beta**n * math.factorial(n) *  (1/sigma1) * (domain_x/sigma1)**(beta-1) * (1/sigma2) * (domain_y/sigma2)**(beta-1) / (1 + (domain_x/sigma1)**beta + (domain_y/sigma2)**beta )**(n+1)
    #print('Pareto: '+str(np.sum(density_tmp)*dx*dy)) # should be 1
    return pd.DataFrame(density, columns=x1[0,:], index=x2[:,0])


# returns the input pdf associated with a given output pdf (Theorem)
## x1: 2-dimensionoal np.meshgrid
## x2: 2-dimensionoal np.meshgrid
## output_pdf: function returning a 2d DataFrame
def input_pdf_from_output_pdf(x1, x2, gamma, b, beta, lambda1, lambda2, output_pdf):
    n=2
    def density(x1,x2):
        with np.errstate(divide='ignore'):
            det = gamma**n * beta**n * b**beta * x1**(beta-1) * x2**(beta-1) / (b**beta + lambda1 * x1**beta + lambda2 * x2**beta)**(n+1)
            det = np.where(np.isinf(det), np.nan, det)
        output = output_pdf(r_i(x1,x2,gamma=gamma, b=b, beta=beta, lambda_i=lambda1, lambda_j=lambda2),
                               r_i(x2,x1,gamma=gamma, b=b, beta=beta, lambda_i=lambda2, lambda_j=lambda1),
                               gamma=gamma)
        return det * np.array(output)
    # Sanity Check: This pdf already integrates to 1 (since the output pdf does) so no normalization needed
    #domain_x, domain_y = np.meshgrid(np.linspace(0,100,10000), np.linspace(0,100,10000))  
    #dx = (np.max(domain_x)-np.min(domain_x))/(np.size(domain_x,1)-1)
    #dy = (np.max(domain_y)-np.min(domain_y))/(np.size(domain_y,0)-1)
    #normalizingConstant = np.sum(density(domain_x,domain_y))*dx*dy
    #print('inputPDF: '+str(normalizingConstant)) # =1
    return pd.DataFrame(density(x1,x2) , columns=x1[0,:], index=x2[:,0])


#############################################
# Output PDFs
#############################################

 #returns density of a uniform distribution with simplex support
def bivariateUniform_pdf(y1, y2, gamma):
    domain_x, domain_y = np.meshgrid(np.linspace(0,gamma,100),np.linspace(0,gamma,100)) #large domain, for integration
    dx = (np.max(domain_x)-np.min(domain_x))/(np.size(domain_x,1)-1)
    dy = (np.max(domain_y)-np.min(domain_y))/(np.size(domain_y,0)-1)
    normalizingConstant = np.sum(domain_x+domain_y<gamma)*dx*dy # uses large range
    density = (y1+y2<gamma)
    return  pd.DataFrame(density/normalizingConstant , columns=y1[0,:], index=y2[:,0])


# returns density of a exponential distribution (arising from linear costs) truncated to simplex support
def bivariateTruncatedExponential_pdf(y1, y2, gamma, kappa=1):
    domain_x, domain_y = np.meshgrid(np.linspace(0,gamma,100),np.linspace(0,gamma,100)) #large domain, for integration
    dx = (np.max(domain_x)-np.min(domain_x))/(np.size(domain_x,1)-1)
    dy = (np.max(domain_y)-np.min(domain_y))/(np.size(domain_y,0)-1)
    normalizingConstant = np.sum(np.exp(-kappa*domain_x-kappa*domain_y)*(domain_x+domain_y<gamma))*dx*dy # uses large range
    
    density = np.exp(-kappa*y1-kappa*y2)*(y1+y2<gamma)
    return  pd.DataFrame(density/normalizingConstant , columns=y1[0,:], index=y2[:,0])
