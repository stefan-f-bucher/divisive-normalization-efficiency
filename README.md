# Code for Bucher & Brandenburger (2022): "Divisive normalization is an efficient code for multivariate Pareto-distributed environments"

The empirical analysis uses the [van Hateren image data set](http://bethgelab.org/datasets/vanhateren/) and the Matlab package [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools).

## Getting Started
Install [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools), [nhist](https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default), and [export_fig](https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig) in the working directory. 

## Numerical Simulations (Python)
- [main.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/main.py) produces figures
  - pdfs.py contains the expressions for the various probability densities
  - mixtureModel.py generates a random sample of Pareto-distributed random variables as a gamma-weighted mixture of independent Weibull random variables
  - plotFunctions.py contains plotting utility functions 
- varianceFormulae_verification.py numerically verifies the result on the Pareto distribution as a mixture model and uses it to generate random samples in order to verify that empirical moments coincide with theoretical ones

## Empirical Analysis of Filtered Image Statistics (Matlab)
- main.m Maximum Likelihood Estimation of Pareto model using filter responses to images from the van Hateren dataset
  - Pareto_pdf.m contains the pdf of the Pareto distribution
  - logLikelihood_Pareto.m computes the log-likelihood under the Pareto distribution
  - logLikelihood_mvtdist.m computes the log-likelihood under the multivariate-t-distribution
  - filterCorrelationHist.m computes the filter responses of a steerable pyramid using function [buildSCFpyr](https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m) from [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools)
