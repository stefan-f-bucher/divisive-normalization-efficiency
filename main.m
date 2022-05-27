% Computes filter responses (scalable pyramid) to  images from the van Hateren dataset using matlabPyrTools (https://github.com/LabForComputationalVision/matlabPyrTools)
% and fits the Pareto model using Maximum Likelihood Estimation

addpath(genpath(pwd));
clear;
close all;

nImages = 5;
feature = 'orientation'; % Compare across 'orientation' or 'scale' 


%% MLE of biavariate Pareto- & t-distributions to the filter responses to images from the van Hateren dataset

parameterEstimates_Pareto = zeros(nImages,7);
parameterEstimates_mvt = zeros(nImages,4);

for i=1:nImages
    image = load('./VanHateren/im'+string(i)+'.mat').oim;
    [c,b,hist_joint, hist_cond, band1, band2] = filterCorrelationHist(image, feature);
    band1abs = abs(band1);
    band2abs = abs(band2);
    data = [band1abs, band2abs];
    datasigned = [band1, band2];

    % Bivariate Maximum Likelihood Estimation of Pareto model (with mu=0)
    %[phat,pci] = mle(data(:), 'logpdf', @(x,sigma1,sigma2,beta) logLikelihood_Pareto(x,[0,0],[sigma1,sigma2],beta,2), 'Start', [ 1,1, 1], 'Options',statset( 'MaxIter',1e5)); 
    [phat,pci] = mle(data(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_Pareto(data,[0,0],[params(1),params(2)],params(3),2), 'Start', [1,1,1], 'Options',statset( 'MaxIter',1e6)); 

    
    parameterEstimates_Pareto(i,1) = 0; %mu1
    parameterEstimates_Pareto(i,2) = 0; %mu2
    parameterEstimates_Pareto(i,3) = phat(1); %sigma1
    parameterEstimates_Pareto(i,4) = phat(2); %sigma2
    parameterEstimates_Pareto(i,5) = phat(3); %beta
    parameterEstimates_Pareto(i,6) = -logLikelihood_Pareto(data(:), [0,0], [phat(1),phat(2)], phat(3), 2); % negative log-likelihood
    parameterEstimates_Pareto(i,7) = 2*3 + parameterEstimates_Pareto(i,6); % AIC = 2*nParams - 2*logLikelihood
    
    % For comparison: Bivariate-t model
    [phat_mvt,pci_mvt] = mle(datasigned(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_mvtdist(data,[1 params(1);params(1) 1],params(2), 2), 'Start', [0.5,1], 'Options',statset( 'MaxIter',1e6)); 
    parameterEstimates_mvt(i,1) = phat_mvt(1); % correlation
    parameterEstimates_mvt(i,2) = phat_mvt(2); % df
    parameterEstimates_mvt(i,3) = -logLikelihood_mvtdist(datasigned(:),[1 phat_mvt(1);phat_mvt(1) 1],phat_mvt(2), 2); % negative log-likelihood
    parameterEstimates_mvt(i,4) = 2*2 + parameterEstimates_mvt(i,3); % AIC = 2*nParams - 2*logLikelihood

end

csvwrite('../figures/parameterEstimates_Pareto.csv',[mean(parameterEstimates_Pareto,1); parameterEstimates_Pareto]); % first row contains mean values
csvwrite('../figures/parameterEstimates_mvt.csv',[mean(parameterEstimates_mvt,1); parameterEstimates_mvt]); % first row contains mean values

figure;
nhist({parameterEstimates_Pareto(:,6),parameterEstimates_mvt(:,3)},'legend',{'Pareto','multivariate-t'},'separate'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default
set(gca,'LooseInset',get(gca,'TightInset'));
saveas(gca,'../figures/parameterHisto_negllh.pdf');


figure;
nhist({parameterEstimates_Pareto(:,7),parameterEstimates_mvt(:,4)},'legend',{'Pareto','multivariate-t'},'separate'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default
set(gca,'LooseInset',get(gca,'TightInset'));
saveas(gca,'../figures/parameterHisto_AIC.pdf');



%% Visualization of Histogram for an example image (last image of loop)

figure;
imagesc(image)
colormap gray

figure;
imagesc(c,b,hist_cond); axis square
colormap gray

figure;
imagesc(c,b,log(hist_joint)); axis square
colormap gray


figure;
subplot(1,2,1);
histogram(band1abs, 51);
subplot(1,2,2);
histogram(band2abs, 51);


%% For Exploration Only: Fitting univariate model to marginal distributions of filter responses 

[phat1d,pci1d] = mle(band1abs, 'logpdf', @(x,mu,sigma,beta) logLikelihood_Pareto(x, mu, sigma, beta, 1), 'Start', [0, 1, 1] );
muhat = phat1d(1)
sigmahat = phat1d(2)
betahat = phat1d(3)
-logLikelihood_Pareto(band1abs, muhat, sigmahat, betahat, 1)

[phat_muzero,pci_muzero] = mle(band1abs, 'logpdf', @(x,sigma,beta) logLikelihood_Pareto(x, 0, sigma, beta, 1), 'Start', [1, 1] );
sigmahat_muzero = phat_muzero(1)
betahat_muzero = phat_muzero(2)
-logLikelihood_Pareto(band1abs, 0, sigmahat_muzero, betahat_muzero, 1)

fit_lognormal = fitdist(band1abs, 'lognormal');
fit_weibull = fitdist(band1abs, 'weibull');
fit_generalizedPareto = fitdist(band1abs, 'generalizedPareto');

fit_lognormal.negloglik
fit_weibull.negloglik
fit_generalizedPareto.negloglik

% Univariate Burr type XII distribution (https://www.mathworks.com/help/stats/burr-type-xii-distribution.html)
% burr(alpha=sigma, c=beta, k=1) corresponds to our univariate Pareto(mu=0, sigma, beta)
%fitdist(band1abs, 'burr'); diverging for some reason...
%pareto = makedist('burr','k',1)
%[phat,pci] = mle(band1abs,'Distribution','burr'); % order of phat: (alpha, c, k)


%distributionFitter(band1abs)
figure;
histfit(band1abs,51,'lognormal');

figure;
H = histogram(band1abs,51,'Normalization','pdf'); hold on;
xs = min(H.BinEdges):0.01:max(H.BinEdges);
prob = zeros(length(xs));
for xind=1:length(xs)
    prob(xind) = Pareto_pdf(xs(xind),muhat, sigmahat, betahat, 1);
end
plot(xs, prob, 'k', 'LineWidth', 2)
