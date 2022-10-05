# Code for [Bucher & Brandenburger (2022): "Divisive normalization is an efficient code for multivariate Pareto-distributed environments"] (https://doi.org/10.1073/pnas.2120581119)

The empirical analysis uses the [van Hateren image data set](http://pirsquared.org/research/vhatdb/full/) (described [here](http://bethgelab.org/datasets/vanhateren/)) and the Matlab package [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools).

## Getting Started
- Install [nhist](https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default) in the working directory. [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools) and [export_fig](https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig) should be included as submodules in externalPackages.
- The MEX files of matlabPyrTools can be compiled by running externalPackages/matlabPyrTools/MEX/compilePyrTools.m (some .c files may only compile after adding "#include <string.h>"). 
- Save [http://pirsquared.org/research/vhatdb/full/vanhateren_iml.zip](http://pirsquared.org/research/vhatdb/full/vanhateren_iml.zip) into a folder and convert to .mat as needed. 

## Numerical Simulations (Python)
- [main.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/main.py) produces figures
  - [pdfs.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/pdfs.py) contains the expressions for the various probability densities
  - [mixtureModel.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/mixtureModel.py) generates a random sample of Pareto-distributed random variables as a gamma-weighted mixture of independent Weibull random variables
  - [plotFunctions.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/plotFunctions.py) contains plotting utility functions 
- [varianceFormulae_verification.py](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/varianceFormulae_verification.py) numerically verifies the result on the Pareto distribution as a mixture model and uses it to generate random samples in order to verify that empirical moments coincide with theoretical ones

## Empirical Analysis of Filtered Image Statistics (Matlab)
- [main.m](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/main.m) Maximum Likelihood Estimation of Pareto model using filter responses to images from the van Hateren dataset
  - [Pareto_pdf.m](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/Pareto_pdf.m) contains the pdf of the Pareto distribution
  - [logLikelihood_Pareto.m](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/logLikelihood_Pareto.m) computes the log-likelihood under the Pareto distribution
  - [logLikelihood_mvtdist.m](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/logLikelihood_mvtdist.m) computes the log-likelihood under the multivariate-t-distribution
  - [filterCorrelationHist.m](https://github.com/stefan-f-bucher/divisive-normalization-efficiency/blob/main/filterCorrelationHist.m) computes the filter responses of a steerable pyramid using function [buildSCFpyr](https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m) from [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools)
