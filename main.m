% Computes filter responses (steerable pyramid) to images from the van Hateren dataset using matlabPyrTools (https://github.com/LabForComputationalVision/matlabPyrTools)
% and fits the Pareto model using Maximum Likelihood Estimation

% Compile MEX files: 
% run externalPackages/matlabPyrTools/MEX/compilePyrTools.m 
% Some .c files may not compile as provided and require an additional "#include <string.h>" in order to compile

addpath(genpath(pwd));
clear;
close all;

runEstimation = 0; % 0: does not run estimation, instead importing existing csv
homogeneousSigma = 1; % 1: Pareto model fitted with restriction sigma1=sigma2.

useRawImages = 1; % 1: use raw images from 'vanHateren_pirsquared' 
imageNumbers = 1:100;

saveFigures = 0; % Note: nhist figures are always saved

%% Setup
if useRawImages
    imagePath = @(i) "./vanHateren_pirsquared/vanhateren_mat/imk"+sprintf('%05d',i)+".mat";
    imageExtension = 'buf'; 
    paramFilePrefix = "parameterEstimatesVanHateren_"; 
    exampleImageNumber = 510;
else
    imagePath = @(i) "./VanHateren_KarklinLewicki/im"+string(i)+".mat"; 
    imageExtension = 'oim';
    paramFilePrefix =  "parameterEstimatesKarklinLewicki_";
    exampleImageNumber = 6; 
end


%% MLE of biavariate Pareto- & t-distributions to the filter responses to images from the van Hateren dataset

if ~ runEstimation
    parameterEstimates_Pareto = csvread(paramFilePrefix+'Pareto.csv',1,0);
    parameterEstimates_mvt = csvread(paramFilePrefix+'mvt.csv',1,0);
else
    parameterEstimates_Pareto = nan(length(imageNumbers),7);
    parameterEstimates_mvt = nan(length(imageNumbers),4);

    for i=1:length(imageNumbers)
        display("image "+string(i))
        image = getfield(load(imagePath(i)),imageExtension);
        [c,b,hist_joint, hist_cond, band1, band2] = filterCorrelationHist(image, 'orientation');
        data_abs = [abs(band1), abs(band2)];
        data_signed = [band1, band2];


        % Bivariate Maximum Likelihood Estimation of multivariate-t model
        lastwarn(''); % Clear last warning message
        [phat_mvt,pci_mvt] = mle(data_signed(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_mvtdist(data,[1 params(1);params(1) 1],params(2), 2), 'Start', [0.5,1], 'Options',statset( 'MaxIter',1e6)); 
        [warnMsg, warnId] = lastwarn;
        if isempty(warnMsg) % only save if there was no warning regarding non-convergence of MLE
            parameterEstimates_mvt(i,1) = phat_mvt(1); % correlation
            parameterEstimates_mvt(i,2) = phat_mvt(2); % df
            parameterEstimates_mvt(i,3) = -logLikelihood_mvtdist(data_signed(:),[1 phat_mvt(1);phat_mvt(1) 1],phat_mvt(2), 2); % negative log-likelihood
            parameterEstimates_mvt(i,4) = 2*2 + 2*parameterEstimates_mvt(i,3); % AIC = 2*nParams - 2*logLikelihood
        else % MLE did not converge
            parameterEstimates_mvt(i,:) = nan;
        end


        % Bivariate Maximum Likelihood Estimation of Pareto model (with mu=0)
        lastwarn(''); % Clear last warning message    
        if homogeneousSigma
            [phat,pci] = mle(data_signed(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_Pareto(data,[0,0],[params(1),params(1)],params(2),2), 'Start', [1,1], 'Options',statset( 'MaxIter',1e6)); 
        else
            [phat,pci] = mle(data_signed(:), 'nloglf', @(params,data,cens,freq) -logLikelihood_Pareto(data,[0,0],[params(1),params(2)],params(3),2), 'Start', [1,1,1], 'Options',statset( 'MaxIter',1e6)); 
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
            parameterEstimates_Pareto(i,6) = -logLikelihood_Pareto(data_signed(:), [0,0], [parameterEstimates_Pareto(i,3),parameterEstimates_Pareto(i,4)], parameterEstimates_Pareto(i,5), 2); % negative log-likelihood
            if homogeneousSigma
                parameterEstimates_Pareto(i,7) = 2*2 + 2*parameterEstimates_Pareto(i,6); % AIC = 2*nParams - 2*logLikelihood
            else
                parameterEstimates_Pareto(i,7) = 2*3 + 2*parameterEstimates_Pareto(i,6); % AIC = 2*nParams - 2*logLikelihood
            end
        else
            parameterEstimates_Pareto(i,:) = nan;
        end
        disp(parameterEstimates_Pareto(i,:))
    end

    csvwrite(paramFilePrefix+'Pareto.csv',[mean(parameterEstimates_Pareto,1,'omitnan'); parameterEstimates_Pareto]); % first row contains mean values
    csvwrite(paramFilePrefix+'mvt.csv',[mean(parameterEstimates_mvt,1,'omitnan'); parameterEstimates_mvt]); % first row contains mean values
end

%% Plotting Goodness-of-Fit and Parameter Estimates

figure;
llhPareto = -parameterEstimates_Pareto(:,6); % log-likelihood
llhMvt =  -parameterEstimates_mvt(:,3); % log-likelihood
schist = scatterhist(llhPareto, llhMvt, 'Direction', 'out', 'Kernel', 'off', 'Color', 'k', 'Marker','.'); 
schist(1).XLim = [-7e06, -3e06];
schist(1).YLim = [-7e06, -3e06];
schist(1).Position = [0.35 0.35 0.55 0.55];
schist(2).Position = [0.35 0.1 0.55 0.25];
schist(3).Position = [0.1 0.35 0.25 0.55];
schist(2).Children.BinWidth = 100000;
schist(3).Children.BinWidth = 100000;
diagline = refline(1,0);
diagline.Color = 'k';
xlabel('llh Pareto');
ylabel('llh multivariate-t');
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 7, 7], 'PaperUnits', 'centimeters', 'PaperSize', [7, 7], 'color','w');
schist(2).YLim = [-0.001e-06, 1.3e-06];
schist(3).YLim = [-0.001e-06, 1.3e-06];
if saveFigures
    print(gcf, '-dpdf', '../figures/llhComparison.pdf');
end

figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5, 5], 'color','w');
histogram(parameterEstimates_Pareto(:,5),0.8:0.02:1.6,'FaceColor','k','FaceAlpha',1); 
xlabel('\beta');
ylim([0,15]);
xticks([0.8,1,1.5]);
ylabel('num. of images');
if saveFigures
    export_fig('../figures/parameterHisto_ParetoBetas.pdf',gcf);
end

figure;
nhist({parameterEstimates_Pareto(:,6),parameterEstimates_mvt(:,3)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','neg. llh','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_negllh.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

%figure;
%nhist({parameterEstimates_Pareto(:,7),parameterEstimates_mvt(:,4)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','AIC','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_AIC.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default




%% Visualization of Histogram for an example image 

exampleImage = getfield(load(imagePath(exampleImageNumber)),imageExtension); 
[c, b, examplehist_joint, examplehist_cond, exampleband1, exampleband2] = filterCorrelationHist(exampleImage, 'orientation');

figure;
set(gcf, 'color','w'); %'Units', 'centimeters', 'Position', [0, 0, 5, 5*size(exampleImage,1)/size(exampleImage,2)], 'PaperUnits', 'centimeters', 'PaperSize', [5*size(exampleImage,1)/size(exampleImage,2), 5], 
if useRawImages
    imagesc(log(exampleImage)); 
else
    imagesc(exampleImage); 
end
axis off;
colormap gray;
if saveFigures
    export_fig('../figures/exampleImage.pdf',gcf);
end

if useRawImages
    lim = 3e3;
else
    lim = 3;
end
ind_b = find(b< -lim, 1, 'last'):find(b> lim, 1, 'first');
ind_c = find(c< -lim, 1, 'last'):find(c> lim, 1, 'first');

figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5.5, 5.5], 'color','w');
imagesc(c(ind_c), b(ind_b), examplehist_cond(ind_b,ind_c)); axis square;
set(gca, 'YDir','normal');
xticks([-lim,0,lim]);
yticks([-lim,0,lim]);
colormap gray;
if saveFigures
    print(gcf, '-dpdf','../figures/examplehistogram_cond.pdf');
end

figure;
set(gcf, 'Units', 'centimeters', 'Position', [0, 0, 5, 5], 'PaperUnits', 'centimeters', 'PaperSize', [5.5, 5.5], 'color','w');
imagesc(c(ind_c), b(ind_b), log(examplehist_joint(ind_b,ind_c))); axis square;
set(gca, 'YDir','normal');
xticks([-lim,0,lim]);
yticks([-lim,0,lim]);
colormap gray;
if saveFigures
    print(gcf, '-dpdf','../figures/examplehistogram_joint.pdf');
end

figure;
subplot(1,2,1);
histogram((exampleband1), 51);
subplot(1,2,2);
histogram((exampleband2), 51);


%% ignore the images where MLE did not converge for one distribution
parameterEstimates_Pareto( any(isnan(parameterEstimates_mvt),2), :) = nan;
parameterEstimates_mvt( any(isnan(parameterEstimates_Pareto),2), :) = nan;

pause(1);
figure;
nhist({parameterEstimates_Pareto(:,6),parameterEstimates_mvt(:,3)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','neg. llh','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_negllh_onlymutual.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

%figure;
%nhist({parameterEstimates_Pareto(:,7),parameterEstimates_mvt(:,4)},'legend',{'Pareto','multivariate-t'},'separate','samebins','stderror','xlabel','AIC','ylabel','num. of images','fsize',20,'eps','../figures/parameterHisto_AIC_onlymutual.eps'); % nhist: https://www.mathworks.com/matlabcentral/fileexchange/27388-plot-and-compare-histograms-pretty-by-default

fid = fopen(paramFilePrefix+'averageonlymutual.txt','wt');
fprintf(fid,'Pareto:\n');
fprintf(fid, '%s\n', mean(parameterEstimates_Pareto,1,'omitnan'));
fprintf(fid,'t-Distribution:\n');
fprintf(fid, '%s\n', mean(parameterEstimates_mvt,1,'omitnan'));
fclose(fid);

