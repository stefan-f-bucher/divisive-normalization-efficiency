% X is m-by-n where m is the number of data points and n the number of dimensions
function llh = logLikelihood_mvtdist(X, C, df, n)
    X = reshape(X, length(X)/n, n);
    %n = size(X,2);
    llh = 0;
    for i=1:size(X,1)
        [~,p] = chol(C);
        if (p==0) && (df>0) % C is symmetric positive definite, df>0
            delta = log(mvtpdf(X(i,:),C,df));
        else
            delta = -Inf;
        end
        if isinf(delta) % mle cannot handle NaN or infinite loglikelihood value
            delta=-1e15;
        end
        llh = llh + delta;
    end
end
