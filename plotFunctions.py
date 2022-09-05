#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides functions producing the plots for the paper.
@author: Stefan Bucher (web@stefan-bucher.ch)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from  matplotlib import gridspec
from matplotlib.widgets import Slider
import matplotlib.ticker as ticker

plt.rcParams['savefig.dpi'] = 900
centimeter = 1/2.54  # centimeters in inches
smallFig = (9*centimeter, 6*centimeter)
mediumFig = (11*centimeter, 11*centimeter)
largeFig = (18*centimeter, 22*centimeter)


############################################
# Univariate Density
############################################
def plotUnivariateDensity(X, density, xlabel, ylabel, xlimlabel=None, ylimlabel=None, fname=None):
    plt.figure(figsize=smallFig)
    ax = plt.gca()
    ax.plot(X, density, 'k')
    ax.set_xlim([0,np.max(X)])
    ax.set_ylim([0,1])
    ax.set_xlabel(xlabel, fontsize=15, labelpad=-10)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=-15)
    ax.set_xticks([0, np.max(X)])
    ax.set_yticks([0, 1])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if xlimlabel is not None:
        ax.set_xticklabels(['0', xlimlabel], fontsize=15)
    if ylimlabel is not None:
        ax.set_yticklabels(['0', ylimlabel], fontsize=15)
    if fname is not None:
        plt.savefig('../figures/' + fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return


############################################
# Joint histogram
############################################
def plotJointHistogram(sample, trunc=5, xlabel=None, ylabel=None, x0label=0, y0label=0, xlimlabel=None, ylimlabel=None, cmap='gray', fname=None):
    plt.figure(figsize=mediumFig)
    h = plt.hist2d(sample[:,0], sample[:,1], bins=100, range=[[0, trunc], [0, trunc]], cmap=cmap, rasterized=True)

    # Labels and Co.
    ax = plt.gca()
    ax.set_xlabel(xlabel, fontsize=18, rotation=0)
    ax.set_xticks([0, trunc])
    if xlimlabel is not None:
        ax.set_xticklabels([x0label, xlimlabel])
    if x0label != 0:
        ax.set_xticklabels([x0label, x0label + '+' + str(int(trunc))])
    for item in ax.get_xticklabels():
        item.set_rotation(0)

    ax.set_ylabel(ylabel, fontsize=18, rotation=0)
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    ax.invert_yaxis()
    ax.set_yticks([0, trunc])
    if ylimlabel is not None:
        ax.set_yticklabels([y0label, ylimlabel])
    if y0label != 0:
        ax.set_yticklabels([y0label, y0label + '+' + str(int(trunc))])
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0,
             verticalalignment="bottom")  # vertical alignment of y tick labels
    ax.tick_params(axis='both', labelsize=18)

    if fname is not None:
            plt.savefig('../figures/'+fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return



############################################
# (Columnwise normalized) conditional histogram ('bow-tie plot')
############################################
def plotConditionalHistogram(sample, trunc=5, xlabel=None, ylabel=None, x0label=0, y0label=0, cmap='gray', fullBowtie=False, fname=None):
    if fullBowtie:
        hist = pd.DataFrame(plt.hist2d(sample[:, 0], sample[:, 1], bins=200, range=[[-trunc, trunc], [-trunc, trunc]], density=True)[0])
        x0label = str(int(-trunc))
        y0label = str(int(-trunc))
        xlimlabel = str(int(trunc))
        ylimlabel = str(int(trunc))
    else:
        hist = pd.DataFrame(plt.hist2d(sample[:,0],sample[:,1], bins=100, range=[[0, trunc], [0, trunc]], density=True)[0])
    conditionalHist = hist.divide(hist.sum(axis=0), axis=1)
    conditionalHist = conditionalHist.divide(conditionalHist.max(axis=0)-conditionalHist.min(axis=0), axis=1)
    conditionalHist = np.where(np.isnan(conditionalHist), 0, conditionalHist)

    plt.figure(figsize=mediumFig)
    plt.imshow(conditionalHist, cmap=cmap, rasterized=True)

    # Labels and Co.
    ax = plt.gca()
    ax.set_xlabel(xlabel, fontsize=18, rotation=0)
    ax.set_ylabel(ylabel, fontsize=18, rotation=0)
    ax.set_xticks([0, hist[0].shape[0]])
    ax.set_yticks([0, hist[0].shape[0]])
    if x0label == 0:
        ax.set_xticklabels([int(trunc), int(trunc)])
    else:
        if fullBowtie:
            ax.set_xticklabels([x0label, xlimlabel])
        else:
            ax.set_xticklabels([x0label, x0label + '+' + str(int(trunc))])
    for item in ax.get_xticklabels():
        item.set_rotation(0)
    if y0label == 0:
        ax.set_yticklabels([int(trunc), int(trunc)])
    else:
        if fullBowtie:
            ax.set_yticklabels([y0label, ylimlabel])
        else:
            ax.set_yticklabels([y0label, y0label + '+' + str(int(trunc))])
    ax.tick_params(axis='both', labelsize=18)
    ax.invert_yaxis()

    if fname is not None:
            plt.savefig('../figures/'+fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return



############################################
# Conditional Density (for 'bow-tie' plots of Figure 2)
############################################

# plots density of Y conditional on X
# X and Y are meshgrid (where X must increase within rows)
def plotConditionalDensity(X, Y, density, xlabel=None, ylabel=None, x0label=0, y0label=0, cmap='gray', fname=None):
    dX = (np.max(X)-np.min(X))/(np.size(X,0)-1)
    dY = (np.max(Y)-np.min(Y))/(np.size(Y,1)-1)
    # Joint probability (density times local area)
    jointProbability = dX * dY * density
    marginalProbability_x = jointProbability.sum(axis=0) #pd.Series
    
    yDensityConditionalOnX = jointProbability.divide(marginalProbability_x, axis=1)
    # rescale conditional histogram column-wise, as is customary in the literature
    yDensityConditionalOnX = yDensityConditionalOnX.divide((yDensityConditionalOnX.max(axis=0) - yDensityConditionalOnX.min(axis=0)), axis=1)
    
    # Plot conditional distribution
    fig, ax = plt.subplots(figsize=mediumFig)
    yDensityConditionalOnX = yDensityConditionalOnX.loc[yDensityConditionalOnX.index<=int(np.max(Y)),yDensityConditionalOnX.columns<=int(np.max(X))]
    sns.heatmap(yDensityConditionalOnX, square=True, cbar=False, cmap=cmap, ax=ax, rasterized=True) # (white for high values)
    
    # Labels and Co.
    ax.set_xlabel(xlabel, fontsize=18, rotation=0)
    ax.set_ylabel(ylabel, fontsize=18, rotation=0)
    ax.set_xticks([0,yDensityConditionalOnX.columns.size])
    ax.set_yticks([0,yDensityConditionalOnX.index.size])
    if x0label==0:
        ax.set_xticklabels([int(np.min(X)),int(np.max(X))]) 
    else:
        ax.set_xticklabels([x0label,x0label+'+'+str(int(np.max(X)))])

    for item in ax.get_xticklabels():
        item.set_rotation(0)

    if y0label ==0:
        ax.set_yticklabels([int(np.min(Y)),int(np.max(Y))])  
    else:
        ax.set_yticklabels([y0label,y0label+'+'+str(int(np.max(Y)))])

    ax.tick_params(axis='both', labelsize=18)
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.02, top=0.98, bottom=0.12, right=0.88) # add a bit of space to the right
    if fname is not None:
        plt.savefig('../figures/'+fname, bbox_inches='tight')
        plt.close('all')
        
    return yDensityConditionalOnX 




############################################
# Joint Density (for Figure 1)
############################################

# plots joint density of X and Y along with its marginals
def plotJointDensity(X, Y, density, xlabel, ylabel, x0label=0, xlimlabel=None, y0label=0, ylimlabel=None, cmap='gray', vmin=None, vmax=None, fname=None):
    fig = plt.figure(figsize=mediumFig)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[1,4],figure=fig)
    gs.update(wspace=0.03, hspace=0.03)
    ax = plt.subplot(gs[1,1])
    axtop = plt.subplot(gs[0,1], sharex=ax)
    axleft =  plt.subplot(gs[1,0], sharey=ax)

    dX = (np.max(X)-np.min(X))/(np.size(X,0)-1)
    dY = (np.max(Y)-np.min(Y))/(np.size(Y,1)-1)
    # Joint probability (density times local area)
    jointProbability = dX * dY * density
    marginalProbability_x = jointProbability.sum(axis=0) 
    marginalProbability_y = jointProbability.sum(axis=1)
    
    # Plot joint distribution
    jointProbability_trunc = jointProbability.loc[jointProbability.index<=int(np.max(Y)), jointProbability.columns<=int(np.max(X))] 
    sns.heatmap(jointProbability_trunc, square=True, cbar=False, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, xticklabels=False, yticklabels=False, rasterized=True)

    # Plot the marginal distributions
    axtop.fill_between(range(len(marginalProbability_x)), 0, marginalProbability_x, color='black') # axis of heatmap is in units of 'pixels'
    axleft.fill_between(marginalProbability_y, 0, range(len(marginalProbability_y)), color='black')

    # Labels and Co.
    axtop.set_ylim([0,axtop.get_ylim()[1]]) 
    axleft.set_xlim(axtop.get_ylim()) # ensure the two marginal pdfs are plotted on the same scale

    ax.set_xlabel(xlabel, fontsize=18, rotation=0)
    ax.set_xticks([0,jointProbability_trunc.columns.size])
    if xlimlabel is not None:
        ax.set_xticklabels([x0label,xlimlabel])
    if x0label != 0:
        ax.set_xticklabels([x0label,x0label+'+'+str(int(np.max(X)))])
    for item in ax.get_xticklabels():#  ,ax.get_yticklabels()]:
        item.set_rotation(0)
        
    ax.set_ylabel(ylabel, fontsize=18, rotation=0)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([0,jointProbability_trunc.index.size])
    if ylimlabel is not None:
        ax.set_yticklabels([y0label,ylimlabel])  
    if y0label != 0:
        ax.set_yticklabels([y0label,y0label+'+'+str(int(np.max(X)))])
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, verticalalignment="bottom") # vertical alignment of y tick labels
    ax.tick_params(axis='both', labelsize=18)

    axtop.get_xaxis().set_visible(False)
    axtop.get_yaxis().set_visible(False)
    axleft.get_xaxis().set_visible(False)
    axleft.get_yaxis().set_visible(False)
    axleft.invert_xaxis()
    #axleft.invert_yaxis() #plot s2 from bottom to top
    #plt.tight_layout()
    plt.subplots_adjust(left=0.02, top=0.98, bottom=0.12, right=0.88) # add a bit of space to the right



    if fname is not None:
            plt.savefig('../figures/'+fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return jointProbability_trunc





############################################
# 3D Plot (e.g. to plot divisive normalization transform)
############################################

# 3D Plotting function
# density can have larger dimensions than X,Y in case the density is needed on all domain to compute the marginals.
# density will be plotted only over X,Y    
def plot3D(X, Y, density, xlabel, ylabel, zlabel, fname=None, show_pdf=True, contourProjections=True, zlim=None, x0label=0, xlimlabel=None, y0label=0, ylimlabel=None, zlimlabel=None, elev=30, azim=28):
    density = np.array(density)
    fig = plt.figure( figsize=mediumFig )
    ax = fig.gca(projection='3d')
    ax.grid(False)
    if show_pdf:
        surf = ax.plot_surface(X,Y,density[0:np.size(X,0),0:np.size(Y,1)],
                            color='grey', alpha=0.6,
                            linewidth=5, antialiased=False,
                            #rstride=8, cstride=8,
                            rcount=50, ccount=50)
        #surf = ax.plot_wireframe(X,Y,density,rcount=25, ccount=25,color='grey',alpha=0.7)
        #ax.contour3D(x1, x2, density)#, 50, cmap='binary')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([0,np.max(X)])
    ax.set_ylim([0,np.max(Y)])
    if zlim:
        ax.set_zlim([0,zlim])
    if contourProjections:
        if show_pdf:
            offset = ax.get_zlim()[1]+0.1
        else:
            offset=0
        # Plot the contours of the density on a z-plane
        cset = ax.contourf(X, Y, density[0:np.size(X,0),0:np.size(Y,1)], zdir='z', levels=18, offset=offset, cmap=cm.coolwarm, alpha=0.9) #cm.Greys
        
        # Plot the marginal densities on the side walls. density is a density where ax0=X and ax1=Y.
        dy = (np.max(Y)-np.min(Y))/(np.size(Y,1)-1)
        marginalDensity_x = np.zeros(X.shape) # marginal density will be stored in one column of this 2d array
        marginalDensity_x[:,-1] = np.sum(density[0:np.size(X,0),:],axis=1)*dy # sum over y-axis 
        cset = ax.contourf(X, Y, marginalDensity_x, zdir='y', offset=0, colors='grey', alpha=0.9)

        dx = (np.max(X)-np.min(X))/(np.size(X,0)-1)
        marginalDensity_y = np.zeros(Y.shape) # marginal density will be stored in one column of this 2d array
        marginalDensity_y[-1,:] = np.sum(density[:,0:np.size(Y,1)],axis=0)*dx # sum over x-axis 
        cset = ax.contourf(X, Y, marginalDensity_y, zdir='x', offset=0, colors='grey', alpha=0.9) #'black'

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18, rotation=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18, rotation=0)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=18, rotation=0)  
    #else:  no z-axis
    #    ax.set_zticks([])
    #    ax.zaxis.line.set_color('lightgrey')
    ax.set_xticks([0,int(np.max(X))]) #.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if xlimlabel is not None:
        ax.set_xticklabels([x0label,xlimlabel])
    if x0label != 0:
        ax.set_xticklabels([x0label,x0label+'+'+str(int(np.max(X)))])
     
    ax.set_yticks([0,int(np.max(Y))]) #.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if ylimlabel is not None:
        ax.set_yticklabels([y0label,ylimlabel])  
    if y0label != 0:
        ax.set_yticklabels([y0label,y0label+'+'+str(int(np.max(X)))])
        
    ax.set_zticks([0,int(zlim)])
    if zlimlabel is not None:
        ax.set_zticklabels([0,zlimlabel])  
    
    ax.tick_params(axis='both', labelsize=18)
    ax.zaxis._axinfo['juggled'] = (1,2,0) #(0,2,1)
    
    if fname is not None:
        plt.savefig('../figures/'+fname, bbox_inches='tight')
        #plt.show()
        
    if show_pdf:    
        return surf
    else:
        return  None





############################################
# Animated Plot for parameter exploration
############################################

# Animated Plot with Sliders for parameters
def animated_Pareto():
    x1, x2 = np.meshgrid(np.linspace(0,2,100),np.linspace(0,2,100))
    density = bivariatePareto_pdf(x2, x1, sigma1=1, sigma2=1, beta=1)
    surf = plot3D(x2, x1, density, xlabel='$x_2$', ylabel='$x_1$', zlabel=None, fname=None, contourProjections=True, zlim=5)
    plt.subplots_adjust(bottom=0.25)

    # Animation
    axsigma1 = plt.axes([0.25, 0.15, 0.65, 0.03])
    sigma1slider = Slider(axsigma1, '$\sigma_1$', 0.1, 5.0, valinit=1, valstep=0.1)
    
    axsigma2 = plt.axes([0.25, 0.1, 0.65, 0.03])
    sigma2slider = Slider(axsigma2, '$\sigma_2$', 0.1, 5.0, valinit=1)
    
    axtopeta = plt.axes([0.25, 0.05, 0.65, 0.03])
    betaslider = Slider(axtopeta, '$beta$', 0.1, 5.0, valinit=1)
    
    plt.show()
    
    def update(val):
        ax.clear()
        #surf = ax.plot_surface(x1, x2, f(x1,x2, sigma1=sigma1slider.val, sigma2=sigma2slider.val, beta=betaslider.val))
        surf = plot3D(x2, x1, bivariatePareto_pdf(x2, x1, sigma1=sigma1slider.val, sigma2=sigma2slider.val, beta=betaslider.val), xlabel='$x_1$', ylabel='$x_2$', zlabel='pdf', fname=None, contourProjections=True, zlim=5)
        fig.canvas.draw_idle()
    
    sigma1slider.on_changed(update)
    sigma2slider.on_changed(update)
    betaslider.on_changed(update)
    return
