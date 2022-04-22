#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script we verify that given a simplex support, the uniform distribution maximizes the entropy.
@author: Stefan Bucher (web@stefan-bucher.ch)
"""
import numpy as np
import pandas as pd
from plotFunctions import plotJointDensity

gamma = 1
nPoints = 100
y1, y2 = np.meshgrid(np.linspace(0, gamma, nPoints), np.linspace(0, gamma, nPoints))

def bivariateUniform_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])

def alt_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)*np.exp(3*(y1+y2))
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])

def alt2_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)*np.exp(3*np.abs(y1-y2)*(y1+y2))
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])

def alt3_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)*np.exp((y1-0.5)**2+(y2-0.5)**2)
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])

def alt4_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)*np.exp(-(y1-0.5)**2-(y2-0.5)**2)
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])


def alt5_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)/((y1+0.01)**2 * (y2+0.01)**2 * (1-y1-y2+0.01)**2)
    density[density>100000]=100000
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])

def alt6_pdf(y1, y2, gamma):
    density = (y1+y2<gamma)*np.exp(-y1-y2)
    normalizingConstant = np.sum(density)
    return pd.DataFrame(density/normalizingConstant, columns=y1[0,:], index=y2[:,0])


def jointEntropy(p):
    h = - p*np.log(p)
    h = h.fillna(0).values
    return np.sum(h)

for pdf in  [alt6_pdf, bivariateUniform_pdf]: #alt_pdf, alt2_pdf,alt3_pdf,alt4_pdf,
    p = pdf(y1, y2, gamma=gamma)
    plotJointDensity(y1, y2, p, xlabel='H='+str(jointEntropy(p)), ylabel=".", xlimlabel='$\gamma$', ylimlabel='$\gamma$')

