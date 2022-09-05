% X is m-by-n where m is the number of data points and n the number of dimensions
function llh = logLikelihood_Pareto(X, mu, sigma, beta, n)
    X = reshape(X, length(X)/n, n);
    %n = size(X,2);
    llh = 0;
    for i=1:size(X,1)
        delta = log(Pareto_pdf(X(i,:), mu, sigma, beta, true));
        if isinf(delta) % mle cannot handle NaN or infinite loglikelihood value
            delta=-1e15;
        end
        llh = llh + delta;
    end
end