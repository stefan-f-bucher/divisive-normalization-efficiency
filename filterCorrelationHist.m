% Function computing histograms of filter correlation
function [c,b,hist_joint, hist_cond, band1, band2] = filterCorrelationHist(image, feature)
    Nsc = 4;
    Nor = 4;
    % Complex-valued steerable pyramid  using Hilbert-transform pairs of
    % filters (Portilla & Simoncelli, Int'l Journal of Computer Vision 2000)
    [pyr0,pind0] = buildSCFpyr(image, Nsc, Nor-1); % image, height, order ( = nbands - 1)
    % Note: buildSCFpyr uses angles = pi*[0:nbands-1]/nbands. So nbands=4
    % implies angles = [pi, 1/4*pi, 1/2*pi, 3/4*pi] rad = [0, 45, 90, 135] deg (counter-clockwise)

    if strcmp(feature,'orientation')
        band1  = real(pyrBand(pyr0,pind0,7)); % access real part of 7th-"finest" subband. Equivalent to real(spyrBand(pyr0, pind0, level=2, band=2)) i.e. 45deg
        band2  = real(pyrBand(pyr0,pind0,9)); % coarser subband (lower spatial freq.). Equivalent to real(spyrBand(pyr0, pind0, level=2, band=4)) i.e. 135deg
    elseif strcmp(feature,'scale')
        band1  = upBlur(real(pyrBand(pyr0,pind0,17)),2);
        band2  = real(pyrBand(pyr0,pind0,9));
    end
    
    [hist_joint,b,c] = jhisto(band1,band2,51); % joint histogram of the two bands with 51 bins
    hist_cond = hist_joint./repmat(max(hist_joint),size(hist_joint,1),1); % conditional histogram (bow-tie plot) 
    
    band1 = band1(:); %serialize 
    band2 = band2(:); 
end 


