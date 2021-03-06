% Computes filter responses (steerable pyramid) to images from the van Hateren dataset using matlabPyrTools (https://github.com/LabForComputationalVision/matlabPyrTools)
% and fits the Pareto model using Maximum Likelihood Estimation

addpath(genpath(pwd));
clear;
close all;

nImages = 100; %50;
feature = 'orientation'; % Compare across 'orientation' or 'scale' 
homogeneousSigma = 1; % 1: Pareto model fitted with restriction sigma1=sigma2.
runEstimation = 0; % 0: does not run estimation, instead importing existing csv

%% MLE of biavariate Pareto- & t-distributions to the filter responses to images from the van Hateren dataset

if ~ runEstimation
    parameterEstimates_Pareto = csvread('parameterEstimates_Pareto.csv',1,0);
    parameterEstimates_mvt = csvread('parameterEstimates_mvt.csv',1,0);
else
    parameterEstimates_Pareto = zeros(nImages,7);
    parameterEstimates_mvt = zeros(nImages,4);

    for i=1:nImages
        image = load('./VanHateren/im'+string(i)+'.mat').oim;
        [c,b,hist_joint, hist_cond, band1, band2] = filterCorrelationHist(image, feature);
        band1abs = abs(band1);
        band2abs = abs(band2);
        data = [band1abs, band2abs];
        datasigned = [band1, band2];


        % Bivariate Maximum Likelihood Estimation of multivariate-t model
        lastwarn(''); % Clear last warning message
        [phat_mvt,pci_mvt] = mle(datasigned(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_mvtdist(data,[1 params(1);params(1) 1],params(2), 2), 'Start', [0.5,1], 'Options',statset( 'MaxIter',1e6)); 
        [warnMsg, warnId] = lastwarn;
        if isempty(warnMsg) % only save if there was no warning regarding non-convergence of MLE
            parameterEstimates_mvt(i,1) = phat_mvt(1); % correlation
            parameterEstimates_mvt(i,2) = phat_mvt(2); % df
            parameterEstimates_mvt(i,3) = -logLikelihood_mvtdist(datasigned(:),[1 phat_mvt(1);phat_mvt(1) 1],phat_mvt(2), 2); % negative log-likelihood
            parameterEstimates_mvt(i,4) = 2*2 + parameterEstimates_mvt(i,3); % AIC = 2*nParams - 2*logLikelihood
        else % MLE did not converge
            parameterEstimates_mvt(i,:) = nan;
            %continue % skips MLE of Pareto where mvt did not converge
        end


        % Bivariate Maximum Likelihood Estimation of Pareto model (with mu=0)
        lastwarn(''); % Clear last warning message    
        if homogeneousSigma
            [phat,pci] = mle(data(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_Pareto(data,[0,0],[params(1),params(1)],params(2),2), 'Start', [1,1], 'Options',statset( 'MaxIter',1e6)); 
        else
            [phat,pci] = mle(data(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_Pareto(data,[0,0],[params(1),params(2)],params(3),2), 'Start', [1,1,1], 'Options',statset( 'MaxIter',1e6)); 
        end
        [warnMsg, warnId] = lastwarn;
        if isempty(warnMsg) % only save if there was no warning regarding non-convergence of MLE
            parameterEstimates_Pareto(i,1) = 0; %mu1
            parameterEstimates_Pareto(i,2) = 0; %mu2
            if homogeneousSigma
                parameterEstimates_Pareto(i,3) = phat(1); %sigma1
                parameterEstimates_Pareto(i,4) = phat(1); %sigma2
                parameterEstimates_Pareto(i,5) = phat(2); %beta
            else
                parameterEstimates_Pareto(i,3) = phat(1); %sigma1
                parameterEstimates_Pareto(i,4) = phat(2); %sigma2
                parameterEstimates_Pareto(i,5) = phat(3); %beta
            end
            parameterEstimates_Pareto(i,6) = -logLikelihood_Pareto(data(:), [0,0], [parameterEstimates_Pareto(i,3),parameterEstimates_Pareto(i,4)], parameterEstimates_Pareto(i,5), 2); % negative log-likelihood
            parameterEstimates_Pareto(i,7) = 2*3 + parameterEstimates_Pareto(i,6); % AIC = 2*nParams - 2*logLikelihood
        else
            parameterEstimates_Pareto(i,:) = nan;
        end
    end

    csvwrite('parameterEstimates_Pareto.csv',[mean(parameterEstimates_Pareto,1,'omitnan'); parameterEstimates_Pareto]); % first row contains mean values
    csvwrite('parameterEstimates_mvt.csv',[mean(parameterEstimates_mvt,1,'omitnan'); parameterEstimates_mvt]); % first row contains mean values
end

%% Plotting Goodness-of-Fit and Parameter Estimates

figure;
llhPareto = -parameterEstimates_Pareto(:,6);
llhMvt =  -parameterEstimates_mvt(:,3);
schist = scatterhist(llhPareto, llhMvt, 'Direction', 'out', 'Kernel', 'off', 'Color', 'k', 'Marker','.'); %'../figures/parameterHisto_negllh.eps'
schist(2).Children.BinWidth = 20000;
schist(3).Children.BinWidth = 20000;
diagline = refline(1,0);
diagline.Color = 'k';
xlabel('llh Pareto');
ylabel('llh multivariate-t');
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 7,7], 'PaperUnits', 'centimeters', 'PaperSize', [7,7], 'color','w');
schist(2).YLim = [-0.0075e-05, 1.05e-05];
schist(3).YLim = [-0.0075e-05, 1.05e-05];
exportgraphics(gcf,'../figures/llhComparison.pdf');


figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5, 5], 'color','w');
histogram(parameterEstimates_Pareto(:,5),0.8:0.02:1.6,'FaceColor','k','FaceAlpha',1); 
xlabel('\beta');
xticks([0.8,1,1.5]);
ylabel('num. of images');
export_fig('../figures/parameterHisto_ParetoBetas.pdf',gcf);

figure;
nhist({parameterEstimates_Pareto(:,6),parameterEstimates_mvt(:,3)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','neg. llh','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_negllh.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

figure;
nhist({parameterEstimates_Pareto(:,7),parameterEstimates_mvt(:,4)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','AIC','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_AIC.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

% ignore the images where MLE did not converge for one distribution
parameterEstimates_Pareto( any(isnan(parameterEstimates_mvt),2), :) = nan;
parameterEstimates_mvt( any(isnan(parameterEstimates_Pareto),2), :) = nan;

figure;
nhist({parameterEstimates_Pareto(:,6),parameterEstimates_mvt(:,3)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','neg. llh','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_negllh_onlymutual.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

figure;
nhist({parameterEstimates_Pareto(:,7),parameterEstimates_mvt(:,4)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','AIC','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_AIC_onlymutual.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

fid = fopen('parameterEstimates_averageonlymutual.txt','wt');
fprintf(fid,'Pareto:\n');
fprintf(fid, '%s\n', mean(parameterEstimates_Pareto,1,'omitnan'));
fprintf(fid,'t-Distribution:\n');
fprintf(fid, '%s\n', mean(parameterEstimates_mvt,1,'omitnan'));
fclose(fid);


%% Visualization of Histogram for an example image 

exampleImage = load('./VanHateren/im6.mat').oim; 
[c,b,examplehist_joint, examplehist_cond, exampleband1, exampleband2] = filterCorrelationHist(exampleImage, feature);


figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5, 5], 'color','w');
imagesc(exampleImage); axis off;
colormap gray
export_fig('../figures/exampleImage.pdf',gcf);

figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5, 5], 'color','w');
imagesc(c,b,examplehist_cond); 
set(gca, 'YDir','normal');
xticks([-3,0,3]);
yticks([-3,0,3]);
colormap gray
export_fig('../figures/examplehistogram_cond.pdf',gcf);


figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5, 5], 'color','w');
imagesc(c,b,log(examplehist_joint)); 
set(gca, 'YDir','normal');
xticks([-3,0,3]);
yticks([-3,0,3]);
colormap gray
export_fig('../figures/examplehistogram_joint.pdf',gcf);


figure;
subplot(1,2,1);
histogram(abs(exampleband1), 51);
subplot(1,2,2);
histogram(abs(exampleband2), 51);
